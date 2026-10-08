// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_mirco_hybrid.hpp"

#ifdef FOUR_C_WITH_MIRCO
#include <Kokkos_Core.hpp>
#endif

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
#include <dlfcn.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <vector>
#endif

FOUR_C_NAMESPACE_OPEN

namespace
{
  [[noreturn]] void abort_hybrid(MPI_Comm comm, const char* message) noexcept
  {
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "MIRCO hybrid execution, rank %d: %s\n", rank, message);
    std::fflush(stderr);
    MPI_Abort(comm, EXIT_FAILURE);
    std::abort();
  }

  void check_mpi(int status, MPI_Comm comm, const char* operation)
  {
    if (status == MPI_SUCCESS) return;
    abort_hybrid(comm, operation);
  }

#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
  enum class WorkerMode
  {
    passive,
    pause,
    bounded
  };

  const char* worker_mode_name(WorkerMode mode)
  {
    switch (mode)
    {
      case WorkerMode::passive:
        return "PASSIVE";
      case WorkerMode::pause:
        return "PAUSE";
      case WorkerMode::bounded:
        return "BOUNDED";
    }
    return "UNKNOWN";
  }

  struct ProcessConfiguration
  {
    // No owned MPI resources: static destruction never calls MPI.
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<int> cpus;
    int rank = -1;
    bool initialized = false;
    bool in_gate = false;
    bool report = false;
    WorkerMode worker_mode = WorkerMode::passive;
    int spin_count = 0;
    int (*pause_resources)(omp_pause_resource_t) = nullptr;
    void (*set_blas_threads)(int) = nullptr;
    int (*get_blas_threads)() = nullptr;
  };

  ProcessConfiguration& configuration()
  {
    static ProcessConfiguration config;
    return config;
  }

  void require(bool condition, const char* message)
  {
    if (!condition) throw std::runtime_error(message);
  }

  [[noreturn]] void throw_system_error(const char* operation)
  {
    const int error = errno;
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(error));
  }

  std::string_view trim(std::string_view value)
  {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
      value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
      value.remove_suffix(1);
    return value;
  }

  int parse_integer(std::string_view value, const char* message)
  {
    value = trim(value);
    int result = -1;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    require(!value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
        message);
    return result;
  }

  std::vector<int> parse_cpus(const char* text)
  {
    require(text != nullptr, "FOUR_C_MIRCO_HYBRID_CPUS is missing.");
    std::string_view remaining(text);
    std::vector<int> result;
    while (true)
    {
      const auto comma = remaining.find(',');
      const int cpu = parse_integer(remaining.substr(0, comma),
          "FOUR_C_MIRCO_HYBRID_CPUS must contain comma-separated integer CPU IDs.");
      require(cpu >= 0 && cpu < CPU_SETSIZE,
          "A requested CPU ID is outside the supported cpu_set_t range.");
      require(std::find(result.begin(), result.end(), cpu) == result.end(),
          "FOUR_C_MIRCO_HYBRID_CPUS contains a repeated CPU ID.");
      result.push_back(cpu);
      if (comma == std::string_view::npos) break;
      remaining.remove_prefix(comma + 1);
    }
    return result;
  }

  bool environment_equals(const char* name, std::string_view expected)
  {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const auto trimmed = trim(value);
    return trimmed.size() == expected.size() &&
           std::equal(trimmed.begin(), trimmed.end(), expected.begin(),
               [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == b; });
  }

  void pin_primary(int cpu)
  {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
      throw_system_error("sched_setaffinity for MIRCO primary thread");
  }

  void prepare_owner()
  {
    auto& config = configuration();
    // libgomp's primary thread belongs to OpenMP place zero even when the MPI
    // phase pinned that thread to this rank's distinct CPU.
    pin_primary(config.cpus.front());
    const int threads = static_cast<int>(config.cpus.size());
    omp_set_num_threads(threads);
    if (config.set_blas_threads != nullptr) config.set_blas_threads(threads);
  }

  void prepare_mpi()
  {
    auto& config = configuration();
    if (config.worker_mode == WorkerMode::pause)
    {
      require(
          omp_get_level() == 0, "MIRCO PAUSE mode cannot release workers inside an OpenMP region.");
      require(config.pause_resources != nullptr && config.pause_resources(omp_pause_soft) == 0,
          "OpenMP soft pause failed. Use FOUR_C_MIRCO_HYBRID_WORKERS=PASSIVE with "
          "OMP_WAIT_POLICY=PASSIVE and GOMP_SPINCOUNT=0, or use a runtime supporting soft pause.");
      omp_set_dynamic(0);
    }
    // Reapply phase controls after pause, which may release the runtime's task/thread state.
    omp_set_num_threads(1);
    if (config.set_blas_threads != nullptr) config.set_blas_threads(1);
    pin_primary(config.cpus.at(config.rank));
    // Kokkos::OpenMP keeps its initialized team size. This changes the OpenMP
    // default and BLAS only; it neither resizes nor finalizes the Kokkos pool.
  }

  double wall_time()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  double process_cpu_time()
  {
    timespec value{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0)
      throw_system_error("clock_gettime(CLOCK_PROCESS_CPUTIME_ID)");
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_nsec) * 1e-9;
  }

  void validate_runtime()
  {
    auto& config = configuration();
    const int threads = static_cast<int>(config.cpus.size());
    const char* omp_threads = std::getenv("OMP_NUM_THREADS");
    require(omp_threads != nullptr &&
                parse_integer(omp_threads, "OMP_NUM_THREADS must be one integer.") == threads,
        "Set OMP_NUM_THREADS to the number of FOUR_C_MIRCO_HYBRID_CPUS.");
    if (config.worker_mode == WorkerMode::pause)
    {
      require(environment_equals("OMP_WAIT_POLICY", "active") ||
                  environment_equals("OMP_WAIT_POLICY", "passive"),
          "Set OMP_WAIT_POLICY=ACTIVE or PASSIVE explicitly for MIRCO PAUSE mode.");
      config.pause_resources = reinterpret_cast<int (*)(omp_pause_resource_t)>(
          dlsym(RTLD_DEFAULT, "omp_pause_resource_all"));
      require(config.pause_resources != nullptr,
          "The loaded OpenMP runtime does not export omp_pause_resource_all. Use "
          "FOUR_C_MIRCO_HYBRID_WORKERS=PASSIVE with OMP_WAIT_POLICY=PASSIVE and GOMP_SPINCOUNT=0.");
    }
    else if (config.worker_mode == WorkerMode::bounded)
    {
      require(environment_equals("OMP_WAIT_POLICY", "active") ||
                  environment_equals("OMP_WAIT_POLICY", "passive"),
          "Set OMP_WAIT_POLICY=ACTIVE or PASSIVE explicitly for MIRCO BOUNDED mode.");
      // GOMP_SPINCOUNT is a GNU runtime setting. Reject other runtimes instead
      // of claiming that their worker spin time has been bounded by this value.
      Dl_info runtime_info{};
      void* omp_symbol = dlsym(RTLD_DEFAULT, "omp_get_max_threads");
      require(
          omp_symbol != nullptr && dladdr(omp_symbol, &runtime_info) != 0 &&
              runtime_info.dli_fname != nullptr &&
              std::string_view(runtime_info.dli_fname).find("libgomp") != std::string_view::npos,
          "MIRCO BOUNDED mode requires the dynamically linked GNU libgomp runtime, which "
          "honors GOMP_SPINCOUNT. Use PASSIVE or PAUSE for another OpenMP runtime.");
    }
    else
    {
      require(environment_equals("OMP_WAIT_POLICY", "passive"),
          "Set OMP_WAIT_POLICY=PASSIVE before starting 4C.");
      require(environment_equals("GOMP_SPINCOUNT", "0"),
          "Set GOMP_SPINCOUNT=0 before starting 4C so idle OpenMP workers sleep.");
    }
    require(
        omp_get_proc_bind() == omp_proc_bind_close, "Set OMP_PROC_BIND=close before starting 4C.");
    require(omp_get_num_places() == threads,
        "Set OMP_PLACES to the ordered singleton places matching FOUR_C_MIRCO_HYBRID_CPUS.");
    for (int i = 0; i < threads; ++i)
    {
      require(omp_get_place_num_procs(i) == 1,
          "MIRCO hybrid execution requires one CPU in each OpenMP place.");
      int cpu = -1;
      omp_get_place_proc_ids(i, &cpu);
      require(cpu == config.cpus[i], "OMP_PLACES order does not match FOUR_C_MIRCO_HYBRID_CPUS.");
    }
    require(Kokkos::is_initialized(), "Initialize Kokkos before MIRCO hybrid execution.");
    require(Kokkos::OpenMP().concurrency() == threads,
        "Kokkos OpenMP concurrency differs from the requested CPU count; initialize it with the "
        "full OMP_NUM_THREADS before enabling hybrid execution.");

    auto set_threads =
        reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    auto get_parallel = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "openblas_get_parallel"));
    auto get_threads = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    if (set_threads != nullptr || get_parallel != nullptr || get_threads != nullptr)
    {
      require(set_threads != nullptr && get_parallel != nullptr && get_threads != nullptr,
          "The loaded OpenBLAS must export its threading-control and threading-query functions.");
      require(get_parallel() == 2,
          "MIRCO hybrid execution requires OpenMP OpenBLAS (openblas_get_parallel() == 2).");
      config.set_blas_threads = set_threads;
      config.get_blas_threads = get_threads;
    }

    omp_set_dynamic(0);
    prepare_owner();
    if (config.get_blas_threads != nullptr)
      require(config.get_blas_threads() == threads,
          "OpenBLAS did not accept the requested full-node thread count.");
    int errors = 0;
    int actual_threads = 0;
#pragma omp parallel num_threads(threads) reduction(+ : errors)
    {
      const int tid = omp_get_thread_num();
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (tid >= threads || sched_getaffinity(0, sizeof(mask), &mask) != 0 ||
          CPU_COUNT(&mask) != 1 || !CPU_ISSET(config.cpus[tid], &mask))
        ++errors;
#pragma omp single
      actual_threads = omp_get_num_threads();
    }
    require(errors == 0 && actual_threads == threads,
        "OpenMP team/affinity verification failed. Use --bind-to none, explicit singleton "
        "OMP_PLACES, OMP_PROC_BIND=close, and CPUs allowed by the job's cpuset.");
    Kokkos::fence("MIRCO hybrid initialization");
    prepare_mpi();
  }
#endif
}  // namespace

bool CONTACT::MircoHybridExecution::enabled()
{
  return std::getenv("FOUR_C_MIRCO_HYBRID_CPUS") != nullptr;
}

void CONTACT::MircoHybridExecution::initialize_process(MPI_Comm comm)
{
  const int local_enabled = enabled() ? 1 : 0;
  int minimum_enabled = 0;
  int maximum_enabled = 0;
  check_mpi(MPI_Allreduce(&local_enabled, &minimum_enabled, 1, MPI_INT, MPI_MIN, comm), comm,
      "MPI_Allreduce(hybrid enable minimum) failed.");
  check_mpi(MPI_Allreduce(&local_enabled, &maximum_enabled, 1, MPI_INT, MPI_MAX, comm), comm,
      "MPI_Allreduce(hybrid enable maximum) failed.");
  if (minimum_enabled != maximum_enabled)
    abort_hybrid(comm, "FOUR_C_MIRCO_HYBRID_CPUS must be set consistently on every rank.");
  if (!local_enabled) return;

#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
  try
  {
    auto& config = configuration();
    require(!config.initialized, "MIRCO hybrid process initialization was called twice.");
    config.comm = comm;
    int ranks = 0;
    check_mpi(MPI_Comm_rank(comm, &config.rank), comm, "MPI_Comm_rank failed.");
    check_mpi(MPI_Comm_size(comm, &ranks), comm, "MPI_Comm_size failed.");
    config.cpus = parse_cpus(std::getenv("FOUR_C_MIRCO_HYBRID_CPUS"));
    require(ranks <= static_cast<int>(config.cpus.size()),
        "MIRCO hybrid execution needs at least one requested CPU for each MPI rank.");
    int cpu_count = static_cast<int>(config.cpus.size());
    check_mpi(MPI_Bcast(&cpu_count, 1, MPI_INT, 0, comm), comm, "MPI_Bcast(CPU count) failed.");
    require(cpu_count == static_cast<int>(config.cpus.size()),
        "FOUR_C_MIRCO_HYBRID_CPUS differs between MPI ranks.");
    auto root_cpus = config.cpus;
    check_mpi(MPI_Bcast(root_cpus.data(), cpu_count, MPI_INT, 0, comm), comm,
        "MPI_Bcast(CPU IDs) failed.");
    require(root_cpus == config.cpus, "FOUR_C_MIRCO_HYBRID_CPUS differs between MPI ranks.");

    MPI_Comm local_comm = MPI_COMM_NULL;
    check_mpi(
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, config.rank, MPI_INFO_NULL, &local_comm),
        comm, "MPI_Comm_split_type failed.");
    int local_ranks = 0;
    check_mpi(MPI_Comm_size(local_comm, &local_ranks), comm, "MPI_Comm_size(shared) failed.");
    check_mpi(MPI_Comm_free(&local_comm), comm, "MPI_Comm_free(shared) failed.");
    require(
        local_ranks == ranks, "MIRCO hybrid execution currently supports one shared-memory node.");

    const char* report_value = std::getenv("FOUR_C_MIRCO_HYBRID_REPORT");
    require(report_value == nullptr || environment_equals("FOUR_C_MIRCO_HYBRID_REPORT", "0") ||
                environment_equals("FOUR_C_MIRCO_HYBRID_REPORT", "1"),
        "FOUR_C_MIRCO_HYBRID_REPORT must be 0 or 1.");
    config.report = environment_equals("FOUR_C_MIRCO_HYBRID_REPORT", "1");
    int root_report = config.report ? 1 : 0;
    check_mpi(MPI_Bcast(&root_report, 1, MPI_INT, 0, comm), comm, "MPI_Bcast(report mode) failed.");
    require(root_report == (config.report ? 1 : 0),
        "FOUR_C_MIRCO_HYBRID_REPORT differs between MPI ranks.");

    const char* worker_mode = std::getenv("FOUR_C_MIRCO_HYBRID_WORKERS");
    require(worker_mode == nullptr ||
                environment_equals("FOUR_C_MIRCO_HYBRID_WORKERS", "passive") ||
                environment_equals("FOUR_C_MIRCO_HYBRID_WORKERS", "pause") ||
                environment_equals("FOUR_C_MIRCO_HYBRID_WORKERS", "bounded"),
        "FOUR_C_MIRCO_HYBRID_WORKERS must be PASSIVE (default), PAUSE, or BOUNDED.");
    config.worker_mode =
        environment_equals("FOUR_C_MIRCO_HYBRID_WORKERS", "pause")
            ? WorkerMode::pause
            : (environment_equals("FOUR_C_MIRCO_HYBRID_WORKERS", "bounded") ? WorkerMode::bounded
                                                                            : WorkerMode::passive);
    int root_mode = static_cast<int>(config.worker_mode);
    check_mpi(MPI_Bcast(&root_mode, 1, MPI_INT, 0, comm), comm, "MPI_Bcast(worker mode) failed.");
    require(root_mode == static_cast<int>(config.worker_mode),
        "FOUR_C_MIRCO_HYBRID_WORKERS differs between MPI ranks.");
    if (config.worker_mode == WorkerMode::bounded)
    {
      const char* spin_count = std::getenv("GOMP_SPINCOUNT");
      require(spin_count != nullptr,
          "MIRCO BOUNDED mode requires explicit GOMP_SPINCOUNT between 0 and 300000.");
      config.spin_count = parse_integer(spin_count,
          "MIRCO BOUNDED mode requires an integer GOMP_SPINCOUNT between 0 and 300000.");
      require(config.spin_count >= 0 && config.spin_count <= 300000,
          "MIRCO BOUNDED mode requires GOMP_SPINCOUNT between 0 and 300000.");
      int root_spin_count = config.spin_count;
      check_mpi(MPI_Bcast(&root_spin_count, 1, MPI_INT, 0, comm), comm,
          "MPI_Bcast(GOMP_SPINCOUNT) failed.");
      require(root_spin_count == config.spin_count, "GOMP_SPINCOUNT differs between MPI ranks.");
    }

    validate_runtime();
    config.initialized = true;
  }
  catch (const std::exception& error)
  {
    abort_hybrid(comm, error.what());
  }
  catch (...)
  {
    abort_hybrid(comm, "Unknown exception during hybrid process initialization.");
  }
#else
  abort_hybrid(comm,
      "FOUR_C_MIRCO_HYBRID_CPUS requires a Linux 4C build with MIRCO and Kokkos OpenMP enabled.");
#endif
}

void CONTACT::MircoHybridExecution::finalize_process()
{
#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
  auto& config = configuration();
  if (!config.initialized) return;
  try
  {
    prepare_owner();
    config.initialized = false;
  }
  catch (const std::exception& error)
  {
    abort_hybrid(config.comm, error.what());
  }
  catch (...)
  {
    abort_hybrid(config.comm, "Unknown exception while restoring the Kokkos finalization team.");
  }
#endif
}

#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
class CONTACT::MircoHybridExecution::Implementation
{
 public:
  explicit Implementation(MPI_Comm comm) : comm_(comm)
  {
    try
    {
      const auto& config = configuration();
      require(config.initialized, "Call MircoHybridExecution::initialize_process() first.");
      require(
          !config.in_gate, "Cannot construct a collective MIRCO gate inside an owner callback.");
      int relation = MPI_UNEQUAL;
      check_mpi(MPI_Comm_compare(comm_, config.comm, &relation), comm_, "MPI_Comm_compare failed.");
      require(relation == MPI_IDENT || relation == MPI_CONGRUENT,
          "MIRCO hybrid execution requires a single MPI group: every world rank must participate "
          "in each interface gate in the same rank order.");
      check_mpi(MPI_Comm_rank(comm_, &rank_), comm_, "MPI_Comm_rank(interface) failed.");
      check_mpi(MPI_Comm_size(comm_, &ranks_), comm_, "MPI_Comm_size(interface) failed.");
      report_ = config.report;
      if (report_ && rank_ == 0) report_values_.resize(static_cast<std::size_t>(ranks_) * 6);

      const auto count = static_cast<std::size_t>(ranks_) + 2;
      require(count <= std::numeric_limits<std::size_t>::max() / sizeof(sem_t),
          "MIRCO hybrid semaphore mapping size overflow.");
      bytes_ = count * sizeof(sem_t);
      require(bytes_ <= static_cast<std::size_t>(std::numeric_limits<off_t>::max()),
          "MIRCO hybrid semaphore mapping exceeds off_t range.");
      const long limit = sysconf(_SC_SEM_VALUE_MAX);
      require(limit <= 0 || ranks_ <= limit, "MPI rank count exceeds semaphore capacity.");

      if (rank_ == 0)
      {
        const char* directory = std::getenv("FOUR_C_MIRCO_HYBRID_TMPDIR");
        if (directory == nullptr) directory = std::getenv("TMPDIR");
        if (directory == nullptr) directory = "/tmp";
        require(
            directory[0] == '/', "The MIRCO hybrid temporary directory must be an absolute path.");
        path_ = std::string(directory) + "/4c-mirco-hybrid-XXXXXX";
        const int fd = mkstemp(path_.data());
        if (fd < 0) throw_system_error("mkstemp for MIRCO hybrid semaphore storage");
        owns_file_ = true;
        if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) throw_system_error("fcntl(FD_CLOEXEC)");
        if (ftruncate(fd, static_cast<off_t>(bytes_)) != 0)
          throw_system_error("ftruncate for MIRCO hybrid semaphore storage");
        map_file(fd);
        if (close(fd) != 0) throw_system_error("close MIRCO hybrid semaphore storage");
        for (std::size_t i = 0; i < count; ++i)
        {
          ::new (static_cast<void*>(semaphores_ + i)) sem_t{};
          if (sem_init(semaphores_ + i, 1, 0) != 0) throw_system_error("sem_init(process-shared)");
        }
      }

      require(path_.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "MIRCO hybrid temporary file path is too long.");
      int path_size = static_cast<int>(path_.size());
      check_mpi(
          MPI_Bcast(&path_size, 1, MPI_INT, 0, comm_), comm_, "MPI_Bcast(path length) failed.");
      path_.resize(path_size);
      check_mpi(MPI_Bcast(path_.data(), path_size, MPI_CHAR, 0, comm_), comm_,
          "MPI_Bcast(semaphore path) failed.");
      // Rank zero initializes all semaphores before broadcasting the path.
      if (rank_ != 0)
      {
        const int fd = open(path_.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) throw_system_error("open MIRCO hybrid semaphore storage");
        map_file(fd);
        if (close(fd) != 0) throw_system_error("close MIRCO hybrid semaphore storage");
      }
      check_mpi(MPI_Barrier(comm_), comm_, "MPI_Barrier(all mappings ready) failed.");
      if (rank_ == 0 && unlink(path_.c_str()) != 0)
        throw_system_error("unlink MIRCO hybrid semaphore storage");
      owns_file_ = false;
      check_mpi(MPI_Barrier(comm_), comm_, "MPI_Barrier(semaphore file unlinked) failed.");
      path_.clear();
    }
    catch (const std::exception& error)
    {
      if (owns_file_) unlink(path_.c_str());
      abort_hybrid(comm_, error.what());
    }
    catch (...)
    {
      if (owns_file_) unlink(path_.c_str());
      abort_hybrid(comm_, "Unknown exception while constructing MIRCO hybrid gate.");
    }
  }

  ~Implementation()
  {
    if (running_) abort_hybrid(comm_, "Cannot destroy a running MIRCO hybrid gate.");
    // No sem_destroy: another process may still be returning from its final
    // completion wait. Unmapping our own view cannot invalidate that process's
    // mapping. Linux unnamed semaphores own no separate kernel resource; the
    // unlinked backing object vanishes when the last mapping is released.
    if (semaphores_ != nullptr && munmap(semaphores_, bytes_) != 0)
      std::fprintf(stderr, "MIRCO hybrid: munmap failed: %s\n", std::strerror(errno));
  }

  void run(const std::function<void()>& work, int interface_id, int owned_nodes)
  {
    try
    {
      require(configuration().initialized, "MIRCO hybrid process is not initialized.");
      require(!running_, "MIRCO hybrid gate callbacks must not be nested or reentrant.");
      require(!configuration().in_gate && omp_in_parallel() == 0,
          "MIRCO hybrid gates must run on the MPI main thread without nested owner callbacks.");
      int main_thread = 0;
      check_mpi(MPI_Is_thread_main(&main_thread), comm_, "MPI_Is_thread_main failed.");
      require(main_thread != 0, "MIRCO hybrid gates must run on the MPI main thread.");
      require(owned_nodes >= 0, "The local MIRCO owned-node count must be nonnegative.");
      running_ = true;
      configuration().in_gate = true;
      ++calls_;
      const double start_wall = wall_time();
      const double start_cpu = process_cpu_time();

      if (configuration().worker_mode == WorkerMode::pause)
      {
        // Small Kokkos operations in the MPI phase can have recreated workers.
        // Every rank must retire those workers before advertising its arrival.
        Kokkos::fence("MIRCO hybrid pause before gate arrival");
        prepare_mpi();
      }
      post(semaphores_);
      if (rank_ == 0)
      {
        // Rank zero also sleeps until every rank has left the previous MPI phase.
        for (int i = 0; i < ranks_; ++i) wait(semaphores_);
      }
      else
        wait(semaphores_ + 2 + rank_);

      const double owner_start_wall = wall_time();
      const double owner_start_cpu = process_cpu_time();
      prepare_owner();
      work();
      Kokkos::fence("MIRCO hybrid owner completion");
      prepare_mpi();
      const double owner_cpu = process_cpu_time() - owner_start_cpu;
      const double owner_wall = wall_time() - owner_start_wall;

      if (rank_ + 1 < ranks_)
      {
        post(semaphores_ + 2 + rank_ + 1);
        wait(semaphores_ + 1);
      }
      else
      {
        // Do not let earlier owners start MPI collectives while a later owner
        // is still using the node. Release everyone only after the final work.
        for (int i = 0; i + 1 < ranks_; ++i) post(semaphores_ + 1);
      }

      const double phase_cpu = process_cpu_time() - start_cpu;
      const double phase_wall = wall_time() - start_wall;
      running_ = false;
      configuration().in_gate = false;
      if (report_)
      {
        const std::array<double, 6> values{owner_wall, phase_wall,
            std::max(0.0, phase_cpu - owner_cpu), static_cast<double>(owned_nodes), owner_cpu,
            phase_cpu};
        // Optional diagnostics only: this collective is after ALL work has
        // finished, outside phase timing, and cannot compete with a MIRCO owner.
        check_mpi(MPI_Gather(
                      values.data(), 6, MPI_DOUBLE, report_values_.data(), 6, MPI_DOUBLE, 0, comm_),
            comm_, "MPI_Gather(MIRCO hybrid timings) failed.");
        if (rank_ == 0) print_report(interface_id);
      }
    }
    catch (const std::exception& error)
    {
      // Unwinding to an outer MPI collective would strand sleeping peers.
      abort_hybrid(comm_, error.what());
    }
    catch (...)
    {
      abort_hybrid(comm_, "Unknown exception in MIRCO hybrid callback or gate.");
    }
  }

 private:
  void map_file(int fd)
  {
    struct stat status{};
    if (fstat(fd, &status) != 0) throw_system_error("fstat MIRCO hybrid semaphore storage");
    require(S_ISREG(status.st_mode) && status.st_size == static_cast<off_t>(bytes_) &&
                status.st_uid == geteuid() && (status.st_mode & (S_IRWXG | S_IRWXO)) == 0,
        "MIRCO hybrid backing file must be a private regular file of the expected size.");
    void* mapping = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) throw_system_error("mmap MIRCO hybrid semaphore storage");
    semaphores_ = static_cast<sem_t*>(mapping);
  }

  static void wait(sem_t* semaphore)
  {
    while (sem_wait(semaphore) != 0)
      if (errno != EINTR) throw_system_error("sem_wait in MIRCO hybrid gate");
  }

  static void post(sem_t* semaphore)
  {
    if (sem_post(semaphore) != 0) throw_system_error("sem_post in MIRCO hybrid gate");
  }

  void print_report(int interface_id) const
  {
    double sum_work = 0.0;
    double max_phase = 0.0;
    double park_cpu = 0.0;
    double active_cpu = 0.0;
    double process_cpu = 0.0;
    long long owned_nodes = 0;
    for (int rank = 0; rank < ranks_; ++rank)
    {
      const auto offset = static_cast<std::size_t>(rank) * 6;
      sum_work += report_values_[offset];
      max_phase = std::max(max_phase, report_values_[offset + 1]);
      park_cpu += report_values_[offset + 2];
      owned_nodes += static_cast<long long>(report_values_[offset + 3]);
      active_cpu += report_values_[offset + 4];
      process_cpu += report_values_[offset + 5];
    }
    std::printf(
        "MIRCO_HYBRID "
        "{\"workers\":\"%s\",\"interface\":%d,\"call\":%llu,\"total_owned_nodes\":%lld,"
        "\"sum_work\":%.9g,\"max_phase\":%.9g,\"park_cpu\":%.9g,"
        "\"active_cpu\":%.9g,\"process_cpu\":%.9g,\"owner_seconds\":[",
        worker_mode_name(configuration().worker_mode), interface_id, calls_, owned_nodes, sum_work,
        max_phase, park_cpu, active_cpu, process_cpu);
    for (int rank = 0; rank < ranks_; ++rank)
      std::printf(
          "%s%.9g", rank == 0 ? "" : ",", report_values_[static_cast<std::size_t>(rank) * 6]);
    std::printf("],\"owner_nodes\":[");
    for (int rank = 0; rank < ranks_; ++rank)
      std::printf("%s%d", rank == 0 ? "" : ",",
          static_cast<int>(report_values_[static_cast<std::size_t>(rank) * 6 + 3]));
    std::printf("]}\n");
    std::fflush(stdout);
  }

  MPI_Comm comm_;
  int rank_ = -1;
  int ranks_ = 0;
  sem_t* semaphores_ = nullptr;
  std::size_t bytes_ = 0;
  std::string path_;
  bool owns_file_ = false;
  bool running_ = false;
  bool report_ = false;
  unsigned long long calls_ = 0;
  std::vector<double> report_values_;
};
#else
class CONTACT::MircoHybridExecution::Implementation
{
};
#endif

CONTACT::MircoHybridExecution::MircoHybridExecution(MPI_Comm comm)
{
  if (!enabled()) return;
#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
  try
  {
    implementation_ = std::make_unique<Implementation>(comm);
  }
  catch (const std::exception& error)
  {
    abort_hybrid(comm, error.what());
  }
  catch (...)
  {
    abort_hybrid(comm, "Unknown exception while allocating MIRCO hybrid gate.");
  }
#else
  abort_hybrid(comm, "MIRCO hybrid execution requires Linux, MIRCO, and Kokkos OpenMP.");
#endif
}

CONTACT::MircoHybridExecution::~MircoHybridExecution() = default;

void CONTACT::MircoHybridExecution::run(
    const std::function<void()>& work, int interface_id, int owned_nodes)
{
#if defined(FOUR_C_WITH_MIRCO) && defined(__linux__) && defined(KOKKOS_ENABLE_OPENMP)
  if (implementation_ != nullptr)
  {
    implementation_->run(work, interface_id, owned_nodes);
    return;
  }
#else
  (void)interface_id;
  (void)owned_nodes;
#endif
  work();
}

FOUR_C_NAMESPACE_CLOSE

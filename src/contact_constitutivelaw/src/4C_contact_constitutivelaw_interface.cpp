
// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_interface.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_contact_constitutivelaw_mirco.hpp"
#include "4C_contact_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_utils_exceptions.hpp"

#include <mirco_kokkostypes.h>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
extern "C"
{
  char* openblas_get_config();
  int openblas_get_num_threads();
  void openblas_set_num_threads(int);
}

#include <optional>

/*
#include <thread>
void sleepy_barrier(MPI_Comm comm)
{
  MPI_Request request;
  MPI_Ibarrier(comm, &request);

  int done = 0;
  while (!done) {
    MPI_Test(&request, &done, MPI_STATUS_IGNORE);

    if (!done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}*/



#include <dirent.h>
#include <sched.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
  void sleepy_barrier(MPI_Comm comm)
  {
    MPI_Request request;
    MPI_Ibarrier(comm, &request);

    int done = 0;
    while (!done)
    {
      MPI_Test(&request, &done, MPI_STATUS_IGNORE);

      if (!done)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  cpu_set_t make_full_node_cpu_set()
  {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    const long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    for (int cpu = 0; cpu < static_cast<int>(num_cpus); ++cpu)
    {
      CPU_SET(cpu, &mask);
    }

    return mask;
  }



  std::vector<pid_t> get_process_thread_ids()
  {
    std::vector<pid_t> tids;

    DIR* dir = opendir("/proc/self/task");
    if (dir == nullptr)
    {
      return tids;
    }

    while (dirent* entry = readdir(dir))
    {
      if (entry->d_name[0] == '.')
      {
        continue;
      }

      tids.push_back(static_cast<pid_t>(std::stoi(entry->d_name)));
    }

    closedir(dir);
    return tids;
  }

  class ScopedAllThreadAffinity
  {
   public:
    explicit ScopedAllThreadAffinity(const cpu_set_t& new_mask)
    {
      const auto tids = get_process_thread_ids();

      for (const pid_t tid : tids)
      {
        cpu_set_t old_mask;
        CPU_ZERO(&old_mask);

        if (sched_getaffinity(tid, sizeof(cpu_set_t), &old_mask) == 0)
        {
          old_masks_.push_back({tid, old_mask});
        }

        if (sched_setaffinity(tid, sizeof(cpu_set_t), &new_mask) != 0)
        {
          std::cerr << "sched_setaffinity failed for tid " << tid << ": " << std::strerror(errno)
                    << '\n';
        }
      }
    }

    ~ScopedAllThreadAffinity()
    {
      for (const auto& [tid, old_mask] : old_masks_)
      {
        sched_setaffinity(tid, sizeof(cpu_set_t), &old_mask);
      }
    }

   private:
    std::vector<std::pair<pid_t, cpu_set_t>> old_masks_;
  };
}  // namespace


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::ConstitutivelawInterface::ConstitutivelawInterface(
    const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData, const int id,
    MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact, bool selfcontact,
    const int contactconstitutivelawid)
    : Interface(interfaceData, id, comm, dim, icontact, selfcontact)
{
  std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw =
      CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::factory(contactconstitutivelawid);
  coconstlaw_ = coconstlaw;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/

void CONTACT::ConstitutivelawInterface::assemble_reg_normal_forces(
    bool& localisincontact, bool& localactivesetchange)
{
  MPI_Barrier(MPI_COMM_WORLD);
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::ConstitutivelawInterface::assemble_reg_normal_forces");

  bool SafeOverSubscription = true;



  MPI_Comm comm_node = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_node);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int node_rank, node_size;
  MPI_Comm_rank(comm_node, &node_rank);
  MPI_Comm_size(comm_node, &node_size);

  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int proc_name_len = 0;
  MPI_Get_processor_name(proc_name, &proc_name_len);


  {
    using LO = int;
    using GO = int;
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;



    using node_type = typename vec_type::node_type;
    using device_type = typename vec_type::device_type;
    using execution_space = typename vec_type::execution_space;
    using memory_space = typename device_type::memory_space;

    if (world_rank == world_size - 1)
    {
      std::cout << "//##################-- Tpetra type information --\n";
      std::cout << "vec_type::node_type        = " << typeid(node_type).name() << '\n';
      std::cout << "vec_type::device_type      = " << typeid(device_type).name() << '\n';
      std::cout << "vec_type::execution_space  = " << typeid(execution_space).name() << '\n';
      std::cout << "vec_type::memory_space     = " << typeid(memory_space).name() << '\n';
      std::cout << '\n';
    }
  }

  if (node_rank == node_size - 1)
  {
    std::cout << "-----------\nSTART assemble_reg_normal_forces()\n";
    std::cout << "\tSafeOverSubscription=" << SafeOverSubscription << "\n";
    std::cout << "-- Kokkos information --\n";
    std::cout << "Threads in use: " << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
    std::cout << "Default execution space: " << typeid(Kokkos::DefaultExecutionSpace).name()
              << "\n";
    std::cout << "Default host execution space: "
              << typeid(Kokkos::DefaultHostExecutionSpace).name() << "\n";
    std::cout << "Default memory space: "
              << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
    std::cout << "Default host memory space: " << typeid(Kokkos::HostSpace).name() << "\n";
    std::cout << "Kokkos num devices = " << Kokkos::num_devices() << "\n";

    std::cout << "world_size=" << world_size << "; world_rank=" << world_rank << "\n";
    std::cout << "node_size=" << node_size << "; node_rank=" << node_rank << "\n";

    std::cout << "node=" << proc_name << " has " << node_size
              << " MPI ranks; representative world_rank=" << world_rank << "\n"
              << "\n";
  }



  double active_time = 0.0;
  double wait_time = 0.0;



  // openblas_set_num_threads(omp_get_max_threads());

#ifdef KOKKOS_ENABLE_OPENMP
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>)
    if (node_size > 1 && SafeOverSubscription)
    {
      if (node_rank == node_size - 1)
        std::cout << "#ifdef KOKKOS_ENABLE_OPENMP if constexpr "
                     "(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) if "
                     "(node_size>1 && SafeOverSubscription) //########## line394\n\n";
    }
#endif

  // TEUCHOS_FUNC_TIME_MONITOR("CONTACT::ConstitutivelawInterface::assemble_reg_normal_forces LOOP
  // PART");
  //  serialize only within each compute node
  if (world_rank == world_size - 1) std::cout << "START LOOP //#line416\n\n";
  for (int owner = 0; owner < node_size; ++owner)
  {
    if (node_rank == owner && source_row_nodes()->num_my_elements() > 0)
    {
      // loop over all source row nodes on the current interface
      for (int i = 0; i < source_row_nodes()->num_my_elements(); ++i)
      {
        const int gid = source_row_nodes()->gid(i);
        Core::Nodes::Node* node = discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %.", gid);
        Node* cnode = dynamic_cast<Node*>(node);

        std::cout << "world_rank=" << world_rank << "; node_rank=" << node_rank
                  << "; node=" << proc_name << "; gid=" << gid << "\n";


        const int dim = cnode->num_dof();
        const double gap = cnode->data().getg();

        const double kappa = cnode->data().kappa();

        double lmuzawan = 0.0;
        for (int k = 0; k < dim; ++k)
          lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

#ifdef CONTACTFDPENALTYKC1
        // set lagrangian multipliers explicitly to constant
        // and corresponding derivatives to zero

        for (int j = 0; j < dim; ++j) cnode->MoData().lm()[j] = i * j;

        cnode->Data().GetDerivZ().clear();

        continue;
#endif

        // Activate/Deactivate node and notice any change
        if ((cnode->active() == false) &&
            (-coconstlaw_->parameter()->get_offset() - kappa * gap >= 0))
        {
          cnode->active() = true;
          localactivesetchange = true;
        }

        else if ((cnode->active() == true) &&
                 (-coconstlaw_->parameter()->get_offset() - kappa * gap < 0))
        {
          cnode->active() = false;
          localactivesetchange = true;

          // std::cout << "node #" << gid << " is now inactive, gap=" << gap << std::endl;
        }
        //********************************************************************

        // Compute derivZ-entries with the Macauley-Bracket
        // of course, this is only done for active constraints in order
        // for linearization and r.h.s to match!
        if (cnode->active() == true)
        {
          double pressure;
          double pressurederiv;
          {
            // Kokkos::fence();

            const cpu_set_t full_node_mask = make_full_node_cpu_set();
            ScopedAllThreadAffinity full_node_affinity(full_node_mask);
            //  Evaluate pressure
            /*const double */ pressure = coconstlaw_->evaluate(kappa * gap, cnode);
            // Evaluate pressure derivative
            /*const double */ pressurederiv = coconstlaw_->evaluate_derivative(kappa * gap, cnode);

            /*cpu_set_t rank_mask2;
        CPU_ZERO(&rank_mask2);
        CPU_SET(node_rank, &rank_mask2);

        DIR* dir2 = opendir("/proc/self/task");
        while (dirent* entry = readdir(dir2)) {
          if (entry->d_name[0] == '.') continue;
          sched_setaffinity(std::atoi(entry->d_name), sizeof(cpu_set_t), &rank_mask2);
        }
        closedir(dir2);*/


            Kokkos::fence();

            openblas_set_num_threads(1);
          }
          localisincontact = true;

          double* normal = cnode->mo_data().n();

          // compute lagrange multipliers and store into node
          for (int j = 0; j < dim; ++j)
            cnode->mo_data().lm()[j] = (lmuzawan - pressure) * normal[j];

          // compute derivatives of lagrange multipliers and store into node
          // contribution of derivative of weighted gap
          std::map<int, double>& derivg = cnode->data().get_deriv_g();
          std::map<int, double>::iterator gcurr;
          // printf("lm=%f\n", -coconstlaw_->evaluate(kappa * gap));

          // contribution of derivative of normal
          std::vector<Core::Gen::Pairedvector<int, double>>& derivn = cnode->data().get_deriv_n();
          Core::Gen::Pairedvector<int, double>::iterator ncurr;

          for (int j = 0; j < dim; ++j)
          {
            for (gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
              cnode->add_deriv_z_value(
                  j, gcurr->first, -kappa * pressurederiv * (gcurr->second) * normal[j]);
            for (ncurr = (derivn[j]).begin(); ncurr != (derivn[j]).end(); ++ncurr)
              cnode->add_deriv_z_value(j, ncurr->first, -pressure * ncurr->second);
            for (ncurr = (derivn[j]).begin(); ncurr != (derivn[j]).end(); ++ncurr)
              cnode->add_deriv_z_value(j, ncurr->first, +lmuzawan * ncurr->second);
          }
        }

        // be sure to remove all LM-related stuff from inactive nodes
        else
        {
          // clear lagrange multipliers
          for (int j = 0; j < dim; ++j) cnode->mo_data().lm()[j] = 0.0;

          // clear derivz
          cnode->data().get_deriv_z().clear();

        }  // Macauley-Bracket
      }  // loop over slave nodes



      // Kokkos::fence();
    }

#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>)
      if (node_size > 1 && SafeOverSubscription)
      {
        const auto barrier_start = std::chrono::steady_clock::now();
        sleepy_barrier(comm_node);
        const auto barrier_end = std::chrono::steady_clock::now();

        wait_time += std::chrono::duration<double>(barrier_end - barrier_start).count();
      }
#endif
  }
  MPI_Comm_free(&comm_node);

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "wait_time=" << wait_time << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::ConstitutivelawInterface::assemble_reg_tangent_forces_penalty()
{
  FOUR_C_THROW("Frictional contact not yet implemented for rough surfaces.");
}

FOUR_C_NAMESPACE_CLOSE

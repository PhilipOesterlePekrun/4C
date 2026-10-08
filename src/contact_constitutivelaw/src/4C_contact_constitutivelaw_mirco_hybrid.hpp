// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_HYBRID_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_HYBRID_HPP

#include "4C_config.hpp"

#include <mpi.h>

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /**
   * Single-node, rank-ordered execution of full-node MIRCO OpenMP work.
   *
   * Opt in with FOUR_C_MIRCO_HYBRID_CPUS (ordered, comma-separated CPU IDs).
   * initialize_process() is collective and must be called on every world rank,
   * after Kokkos initialization, even when disabled. FOUR_C_MIRCO_HYBRID_WORKERS
   * selects PASSIVE (default: OMP_WAIT_POLICY=PASSIVE and GOMP_SPINCOUNT=0),
   * PAUSE (explicit soft pauses retire workers between owner batches, allowing
   * OMP_WAIT_POLICY=ACTIVE during the batch), or BOUNDED (GNU libgomp retains
   * workers and spins for an explicitly bounded GOMP_SPINCOUNT of 0..300000
   * iterations before sleeping; OMP_WAIT_POLICY must be ACTIVE or PASSIVE).
   * BOUNDED permits brief CPU usage while workers finish their spin interval;
   * a spin-count limit is not a wall-time guarantee. PAUSE is opt-in and may destroy
   * worker threads: code must not require persistent OpenMP threadprivate/TLS
   * state across batches. All explicit OpenMP tasks must have completed before
   * entering a gate or returning from its callback. finalize_process() restores
   * the full team before Kokkos finalization. Neither function reinitializes Kokkos.
   *
   * Construction and run() are collective on a communicator identical or
   * congruent to that passed to initialize_process(). All ranks, including ranks
   * owning no nodes, must call run() in the same order. No callback may communicate
   * with a sleeping rank, and no MPI request requiring peer progress may remain
   * outstanding when entering run(). The callback executes on the MPI main thread.
   *
   * Every callback has finished before any rank returns. Destruction only unmaps
   * this process's storage, performs no MPI operations, and must happen after the
   * last local run() has returned. The borrowed communicator must outlive this object.
   */
  class MircoHybridExecution
  {
   public:
    static bool enabled();
    static void initialize_process(MPI_Comm comm);
    static void finalize_process();

    explicit MircoHybridExecution(MPI_Comm comm);
    ~MircoHybridExecution();

    MircoHybridExecution(const MircoHybridExecution&) = delete;
    MircoHybridExecution& operator=(const MircoHybridExecution&) = delete;
    MircoHybridExecution(MircoHybridExecution&&) = delete;
    MircoHybridExecution& operator=(MircoHybridExecution&&) = delete;

    void run(const std::function<void()>& work, int interface_id, int owned_nodes);

   private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif

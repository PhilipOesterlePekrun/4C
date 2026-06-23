
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

#include <quo.h>

#include <Kokkos_Core.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <thread>
#include <chrono>

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
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
}

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
  TEUCHOS_FUNC_TIME_MONITOR("CONTACT::ConstitutivelawInterface::assemble_reg_normal_forces");

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  char proc_name[MPI_MAX_PROCESSOR_NAME];
  int proc_name_len = 0;
  MPI_Get_processor_name(proc_name, &proc_name_len);

  QUO_context quo = nullptr;

  if (QUO_SUCCESS != QUO_create(&quo, MPI_COMM_WORLD))
  {
    //std::cerr << "QUO_create failed on world_rank=" << world_rank << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int node_rank = 0;
  int node_size = 0;

  QUO_id(quo, &node_rank);
  QUO_nqids(quo, &node_size);

  if (world_rank == 0)
  {
    using LO = int;
    using GO = int;
    using map_type = Tpetra::Map<LO, GO>;
    using vec_type = Tpetra::Vector<double, LO, GO>;

    using node_type = typename vec_type::node_type;
    using device_type = typename vec_type::device_type;
    using execution_space = typename vec_type::execution_space;
    using memory_space = typename device_type::memory_space;

    std::cout << "-----------\n";
    std::cout << "START assemble_reg_normal_forces()\n";

    std::cout << "-- MPI / QUO information --\n";
    std::cout << "world_size = " << world_size << "\n";
    std::cout << "node_size  = " << node_size << "\n";
    std::cout << "node       = " << proc_name << "\n\n";

    std::cout << "-- Kokkos information --\n";
    std::cout << "Threads in use: " << Kokkos::DefaultExecutionSpace().concurrency() << "\n";
    std::cout << "Default execution space: "
              << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
    std::cout << "Default host execution space: "
              << typeid(Kokkos::DefaultHostExecutionSpace).name() << "\n";
    std::cout << "Default memory space: "
              << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
    std::cout << "Default host memory space: "
              << typeid(Kokkos::HostSpace).name() << "\n";
    std::cout << "Kokkos num devices = " << Kokkos::num_devices() << "\n\n";

    std::cout << "-- Tpetra type information --\n";
    std::cout << "vec_type::node_type        = " << typeid(node_type).name() << "\n";
    std::cout << "vec_type::device_type      = " << typeid(device_type).name() << "\n";
    std::cout << "vec_type::execution_space  = " << typeid(execution_space).name() << "\n";
    std::cout << "vec_type::memory_space     = " << typeid(memory_space).name() << "\n\n";
  }

  const int num_source_nodes = source_row_nodes()->num_my_elements();

  for (int owner = 0; owner < node_size; ++owner)
  {
    if (node_rank == owner && num_source_nodes > 0)
    {
      char* before_binding = nullptr;
      char* pushed_binding = nullptr;
      char* after_binding = nullptr;

      QUO_stringify_cbind(quo, &before_binding);

      if (QUO_SUCCESS != QUO_bind_push(quo, QUO_BIND_PUSH_OBJ, QUO_OBJ_MACHINE, -1))
      {
        //std::cerr << "QUO_bind_push failed on world_rank=" << world_rank
          //        << ", node_rank=" << node_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
      }

      QUO_stringify_cbind(quo, &pushed_binding);

      /*std::cout << "QUO active rank: world_rank=" << world_rank
                << ", node_rank=" << node_rank
                << ", node=" << proc_name
                << ", source_nodes=" << num_source_nodes << "\n";*/
      std::cout << "  before: " << (before_binding ? before_binding : "null") << "\n";
      std::cout << "  pushed: " << (pushed_binding ? pushed_binding : "null") << "\n";

      for (int i = 0; i < num_source_nodes; ++i)
      {
        const int gid = source_row_nodes()->gid(i);

        Core::Nodes::Node* node = discret().g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %.", gid);

        Node* cnode = dynamic_cast<Node*>(node);
        if (!cnode) FOUR_C_THROW("Cannot cast node with gid % to contact node.", gid);

        const int dim = cnode->num_dof();
        const double gap = cnode->data().getg();
        const double kappa = cnode->data().kappa();

        double lmuzawan = 0.0;
        for (int k = 0; k < dim; ++k)
          lmuzawan += cnode->mo_data().lmuzawa()[k] * cnode->mo_data().n()[k];

#ifdef CONTACTFDPENALTYKC1
        for (int j = 0; j < dim; ++j) cnode->MoData().lm()[j] = i * j;

        cnode->Data().GetDerivZ().clear();

        continue;
#endif

        if (!cnode->active() &&
            (-coconstlaw_->parameter()->get_offset() - kappa * gap >= 0))
        {
          cnode->active() = true;
          localactivesetchange = true;
        }
        else if (cnode->active() &&
                 (-coconstlaw_->parameter()->get_offset() - kappa * gap < 0))
        {
          cnode->active() = false;
          localactivesetchange = true;
        }

        if (cnode->active())
        {
          const double pressure = coconstlaw_->evaluate(kappa * gap, cnode);
          const double pressurederiv = coconstlaw_->evaluate_derivative(kappa * gap, cnode);

          localisincontact = true;

          double* normal = cnode->mo_data().n();

          for (int j = 0; j < dim; ++j)
            cnode->mo_data().lm()[j] = (lmuzawan - pressure) * normal[j];

          std::map<int, double>& derivg = cnode->data().get_deriv_g();
          std::vector<Core::Gen::Pairedvector<int, double>>& derivn =
              cnode->data().get_deriv_n();

          for (int j = 0; j < dim; ++j)
          {
            for (auto gcurr = derivg.begin(); gcurr != derivg.end(); ++gcurr)
              cnode->add_deriv_z_value(
                  j, gcurr->first, -kappa * pressurederiv * gcurr->second * normal[j]);

            for (auto ncurr = derivn[j].begin(); ncurr != derivn[j].end(); ++ncurr)
              cnode->add_deriv_z_value(j, ncurr->first, -pressure * ncurr->second);

            for (auto ncurr = derivn[j].begin(); ncurr != derivn[j].end(); ++ncurr)
              cnode->add_deriv_z_value(j, ncurr->first, +lmuzawan * ncurr->second);
          }
        }
        else
        {
          for (int j = 0; j < dim; ++j) cnode->mo_data().lm()[j] = 0.0;

          cnode->data().get_deriv_z().clear();
        }
      }

      Kokkos::fence();

      if (QUO_SUCCESS != QUO_bind_pop(quo))
      {
        //std::cerr << "QUO_bind_pop failed on world_rank=" << world_rank
          //        << ", node_rank=" << node_rank << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
      }

      QUO_stringify_cbind(quo, &after_binding);

      std::cout << "  after:  " << (after_binding ? after_binding : "null") << "\n\n";

      std::free(before_binding);
      std::free(pushed_binding);
      std::free(after_binding);
    }

    /*if (QUO_SUCCESS != QUO_barrier(quo))
    {
      //std::cerr << "QUO_barrier failed on world_rank=" << world_rank
        //        << ", node_rank=" << node_rank << "\n";
      MPI_Abort(MPI_COMM_WORLD, 4);
    }*/
    sleepy_barrier(MPI_COMM_WORLD);
  }

  if (QUO_SUCCESS != QUO_free(quo))
  {
    //std::cerr << "QUO_free failed on world_rank=" << world_rank << "\n";
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::ConstitutivelawInterface::assemble_reg_tangent_forces_penalty()
{
  FOUR_C_THROW("Frictional contact not yet implemented for rough surfaces.");
}

FOUR_C_NAMESPACE_CLOSE

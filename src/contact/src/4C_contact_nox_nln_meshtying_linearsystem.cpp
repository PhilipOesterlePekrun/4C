// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nox_nln_meshtying_linearsystem.hpp"  // base class

#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_input.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

//#/# Needed for the muelu precond:
#include "4C_linear_solver_method.hpp"
#include "4C_global_data.hpp"
#include "4C_contact_meshtying_abstract_strategy.hpp" //# or higher
#include "4C_fem_discretization.hpp"
//#/#

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::MeshTying::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector, scalingObject),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
std::cout<<"NOX::Nln::MeshTying::LinearSystem() line39 //#\n";
  
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::MeshTying::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec)
{
std::cout<<"NOX::Nln::MeshTying::LinearSystem() line59 //#\n";
  
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SolverParams NOX::Nln::MeshTying::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;

  bool isAdaptiveControl = p.get<bool>("Adaptive Control");
  double adaptiveControlObjective = p.get<double>("Adaptive Control Objective");
  // This value is specified in the underlying time integrator
  // (i.e. RunPreNoxNlnSolve())
  int step = p.get<int>("Current Time Step");
  // This value is specified in the PrePostOperator object of
  // the non-linear solver (i.e. runPreIterate())
  int nlnIter = p.get<int>("Number of Nonlinear Iterations");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    const auto iNlnReq = std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    FOUR_C_ASSERT(iNlnReq,
        "NOX::Nln::MeshTying::LinearSystem::set_solver_options(): required interface cast "
        "failed");

    double worst = iNlnReq->calc_ref_norm_force();
    // This value has to be specified in the PrePostOperator object of
    // the non-linear solver (i.e. runPreSolve())
    double wanted = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_tolerance = wanted;
    solver_params.nonlin_residual = worst;
    solver_params.lin_tol_better = adaptiveControlObjective;
  }
std::cout<<"NOX::Nln::MeshTying::LinearSystem::set_solver_options() line93 //#\n";

  // nothing more to do for a pure structural solver
  if (solverType == NOX::Nln::sol_structure) return solver_params;
std::cout<<"NOX::Nln::MeshTying::LinearSystem::set_solver_options() line96 //#\n";
  // update information about active slave dofs
  // ---------------------------------------------------------------------
  // feed solver/preconditioner with additional information about the
  // contact/meshtying problem
  // ---------------------------------------------------------------------
  {
    // TODO: maps for merged meshtying and contact problem !!!
    // feed Belos based solvers with contact information
    if (solverPtr->params().isSublist("Belos Parameters"))
    {
      //#/# FEB 2026; HERE 100%
      if (i_constr_prec_.size() > 1)
        FOUR_C_THROW(
            "Currently only one constraint preconditioner interface can be handled! \n "
            "Needs to be extended!");

      Teuchos::ParameterList& mueluParams = solverPtr->params().sublist("Belos Parameters");

      // vector entries:
      // (0) masterDofMap
      // (1) slaveDofMap
      // (2) innerDofMap
      // (3) activeDofMap
      std::vector<Teuchos::RCP<Core::LinAlg::Map>> prec_maps(4, Teuchos::null);
      i_constr_prec_.begin()->second->fill_maps_for_preconditioner(prec_maps); //#/# Holy shit it seems second is actually a `CONTACT::MtAbstractStrategy` object
      
      //auto& abstractStrat = static_cast<CONTACT::MtAbstractStrategy&>(*i_constr_prec_.begin()->second);
                                        
          std::shared_ptr<Mortar::StrategyBase> strategy =
                  std::dynamic_pointer_cast<Mortar::StrategyBase>(
                      Core::Utils::shared_ptr_from_ref(*i_constr_prec_.begin()->second));//abstractStrat));
      
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact masterDofMap", Teuchos::rcpFromRef((prec_maps[0]->get_epetra_map())));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact slaveDofMap", Teuchos::rcpFromRef(prec_maps[1]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact innerDofMap", Teuchos::rcpFromRef(prec_maps[2]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact activeDofMap", Teuchos::rcpFromRef(prec_maps[3]->get_epetra_map()));
      // contact or contact/meshtying
      if (i_constr_prec_.begin()->first == NOX::Nln::sol_contact)
        mueluParams.set<std::string>("Core::ProblemType", "contact");
      // only meshtying
      else if (i_constr_prec_.begin()->first == NOX::Nln::sol_meshtying)
        mueluParams.set<std::string>("Core::ProblemType", "meshtying");
      else
        FOUR_C_THROW("Currently we support only a pure meshtying OR a pure contact problem!");
      
      // construct the mapping of the dual node IDs to primal node IDs
      std::shared_ptr<Core::FE::Discretization> discret =
      Global::Problem::instance()->get_dis("structure"); //#
      
      std::shared_ptr<std::map<int, int>> dual2primal_map = std::make_shared<std::map<int, int>>();
      const std::shared_ptr<const Core::LinAlg::Map> gs_node_row_map =
          strategy->slave_row_nodes_ptr();
      const Core::LinAlg::Map* solid_node_map = discret->node_row_map();
      for (int dual_lid = 0; dual_lid < gs_node_row_map->num_my_elements(); dual_lid++)
      {
        int dual_gid = gs_node_row_map->gid(dual_lid);
        if (discret->have_global_node(dual_gid))
          (*dual2primal_map)[dual_lid] = solid_node_map->lid(dual_gid);
      }
      mueluParams.set<Teuchos::RCP<std::map<int, int>>>(
          "Interface DualNodeID to PrimalNodeID", dual2primal_map);

//# .set("reuse") is missing here?
      mueluParams.set<int>("time step", step);
      // increase counter by one (historical reasons)
      mueluParams.set<int>("iter", nlnIter + 1);
      
      
      // specific cases //#
      
      const Teuchos::ParameterList& mcparams = Global::Problem::instance()->contact_dynamic_params();

    const auto sys_type = Teuchos::getIntegralValue<CONTACT::SystemType>(mcparams, "SYSTEM");

      if(sys_type==CONTACT::SystemType::saddlepoint) //# Or just use i_constr_prec_.begin()->second->is_saddle_point_system()
      {
        const auto sol_type =
            Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(mcparams, "STRATEGY");
        if (sol_type == CONTACT::SolvingStrategy::lagmult) //# there exists a `class LAGPENCONSTRAINT::NoxInterfacePrec : public NOX::Nln::CONSTRAINT::Interface::Preconditioner`, and `i_constr_prec_.begin()->second` is of type `NOX::Nln::CONSTRAINT::Interface::Preconditioner`; We could try to cast and check if null? Though there is probably a better way
        {
          // provide null space information
          const int lin_solver_id = mcparams.get<int>("LINEAR_SOLVER");
          const auto prec = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
              Global::Problem::instance()->solver_params(lin_solver_id), "AZPREC");
          if (prec == Core::LinearSolver::PreconditionerType::multigrid_muelu)
          {
            // feed Belos based solvers with contact information
            if (solverPtr->params().isSublist("Belos Parameters"))
            {


              // compute the nullspace vectors for the Lagrange multiplier field for MueLu
                if (solverPtr->params().isSublist("Belos Parameters") and
                    solverPtr->params().isSublist("MueLu Parameters"))
                {
                  int dim_nullspace = 3;//discretization()->n_dim();

                  // get the degree of freedom map from the block matrix
                  auto block_mat_blocked_operator =
            std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
                Core::Utils::shared_ptr_from_ref(jacobian_ptr())); //# or slightly different?

                  if (!block_mat_blocked_operator)
                    FOUR_C_THROW("Failed to cast blockMat to BlockSparseMatrixBase");

                  auto mat11 = block_mat_blocked_operator->matrix(1, 1);
                  const Core::LinAlg::Map& dofmap = mat11.domain_map();

                  // set the nullspace
                  std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
                      std::make_shared<Core::LinAlg::MultiVector<double>>(dofmap, dim_nullspace, true);
                  for (int ldof = 0; ldof < dofmap.num_my_elements(); ++ldof)
                  {
                    nullspace->replace_local_value(ldof, ldof % dim_nullspace, 1.0);
                  }

                  // add the nullspace to the parameter list
                  solverPtr->params()
                      .sublist("Inverse2")
                      .sublist("MueLu Parameters")
                      .set("nullspace", nullspace);
                }

              std::cout << "nln_meshtying line218; solverPtr->params().print() //#\n";
              solverPtr->params().print();
            }
          }
        }
      }
      
      
      
    }
  }  // end: feed solver with contact/meshtying information

  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::MeshTying::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  currSolver = solvers.at(NOX::Nln::sol_meshtying);
  return NOX::Nln::sol_meshtying;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::MeshTying::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::CONTACT::LinearSystem::" << functionName << " - " << errorMsg
                 << std::endl;
  }
  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE

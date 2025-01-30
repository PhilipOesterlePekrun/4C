// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_inpar_mpc_rve.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN
// set the mpc specific parameters
void Inpar::RveMpc::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;


  Teuchos::ParameterList& mpc = list.sublist("MULTI POINT CONSTRAINTS", false, "");

  Teuchos::setStringToIntegralParameter<Inpar::RveMpc::RveReferenceDeformationDefinition>(
      "RVE_REFERENCE_POINTS", "automatic", "Method of definition of the reference points of an RVE",
      Teuchos::tuple<std::string>("automatic", "manual"),
      Teuchos::tuple<Inpar::RveMpc::RveReferenceDeformationDefinition>(automatic, manual), &mpc);

  Teuchos::setStringToIntegralParameter<Inpar::RveMpc::EnforcementStrategy>("ENFORCEMENT",
      "penalty_method", "Method to enforce the multi point constraint",
      Teuchos::tuple<std::string>("penalty_method", "lagrange_multiplier_method"),
      Teuchos::tuple<Inpar::RveMpc::EnforcementStrategy>(penalty, lagrangeMultiplier), &mpc);

  Teuchos::setDoubleParameter("PENALTY_PARAM", 1e5, "Value of the penalty parameter", &mpc);
}

// set mpc specific conditions
void Inpar::RveMpc::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  // ================================================================================================
  Core::Conditions::ConditionDefinition rve_lineperiodic_condition(
      "DESIGN LINE PERIODIC RVE 2D BOUNDARY CONDITIONS", "LinePeriodicRve",
      "definition of edges forming 2D periodic boundary conditions",
      Core::Conditions::LineRvePeriodic, false, Core::Conditions::geometry_type_line);

  rve_lineperiodic_condition.add_component(
      selection<std::string>("EDGE", {"x+", "x-", "y+", "y-", "undefined"},
          {.description = "edge line id", .default_value = "undefined"}));

  condlist.push_back(rve_lineperiodic_condition);

  // ================================================================================================
  Core::Conditions::ConditionDefinition rve_surfperiodic_condition(
      "DESIGN SURF PERIODIC RVE 3D BOUNDARY CONDITIONS", "SurfacePeriodicRve",
      "definition of surfaces forming 3D periodic boundary conditions",
      Core::Conditions::SurfaceRvePeriodic, false, Core::Conditions::geometry_type_surface);

  rve_surfperiodic_condition.add_component(
      selection<std::string>("SURF", {"x+", "x-", "y+", "y-", "z+", "z-", "undefined"},
          {.description = "surface id", .default_value = "undefined"}));

  condlist.push_back(rve_surfperiodic_condition);

  // ================================================================================================
  Core::Conditions::ConditionDefinition rve_cornerpoint_condition(
      "DESIGN POINT PERIODIC RVE 2D BOUNDARY REFERENCE CONDITIONS", "PointPeriodicRveReferenceNode",
      "definition of reference points defining the reference vector of the periodic boundary"
      "condition -  only required if RVE_REFERENCE_POINTS = automatic",
      Core::Conditions::PointRvePeriodicReference, false, Core::Conditions::geometry_type_point);

  rve_cornerpoint_condition.add_component(
      selection<std::string>("POSITION", {"N1L", "N1B", "N2", "N4", "N1", "N3", "undefined"},
          {.description = "position of reference node", .default_value = "undefined"}));

  condlist.push_back(rve_cornerpoint_condition);

  // ================================================================================================
  Core::Conditions::ConditionDefinition linear_ce("DESIGN POINT COUPLED DOF EQUATION CONDITIONS",
      "PointLinearCoupledEquation",
      "definition of the term of a linear couple equation coupling different degrees of "
      "freedom in "
      "2d",
      Core::Conditions::PointLinearCoupledEquation, false, Core::Conditions::geometry_type_point);

  linear_ce.add_component(entry<int>("EQUATION", {.description = "EQUATION"}));
  linear_ce.add_component(selection<std::string>("ADD", {"dispx", "dispy", "undefined"},
      {.description = "degrees of freedom", .default_value = "undefined"}));
  linear_ce.add_component(entry<double>("COEFFICIENT"));

  condlist.push_back(linear_ce);
  /*--------------------------------------------------------------------*/
}
FOUR_C_NAMESPACE_CLOSE
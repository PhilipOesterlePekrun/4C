// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_ale.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::ALE::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& adyn = list.sublist("ALE DYNAMIC", false, "");

  Core::Utils::double_parameter("TIMESTEP", 0.1, "time step size", &adyn);
  Core::Utils::int_parameter("NUMSTEP", 41, "max number of time steps", &adyn);
  Core::Utils::double_parameter("MAXTIME", 4.0, "max simulation time", &adyn);

  setStringToIntegralParameter<Inpar::ALE::AleDynamic>("ALE_TYPE", "solid",
      "ale mesh movement algorithm",
      tuple<std::string>("solid", "solid_linear", "laplace_material", "laplace_spatial",
          "springs_material", "springs_spatial"),
      tuple<Inpar::ALE::AleDynamic>(solid, solid_linear, laplace_material, laplace_spatial,
          springs_material, springs_spatial),
      &adyn);

  Core::Utils::bool_parameter("ASSESSMESHQUALITY", "no",
      "Evaluate element quality measure according to [Oddy et al. 1988]", &adyn);

  Core::Utils::bool_parameter("UPDATEMATRIX", "no",
      "Update stiffness matrix in every time step (only for linear/material strategies)", &adyn);

  Core::Utils::int_parameter("MAXITER", 1, "Maximum number of newton iterations.", &adyn);
  Core::Utils::double_parameter(
      "TOLRES", 1.0e-06, "Absolute tolerance for length scaled L2 residual norm ", &adyn);
  Core::Utils::double_parameter(
      "TOLDISP", 1.0e-06, "Absolute tolerance for length scaled L2 increment norm ", &adyn);

  Core::Utils::int_parameter("NUM_INITSTEP", 0, "", &adyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart data every RESTARTEVERY steps", &adyn);
  Core::Utils::int_parameter("RESULTSEVERY", 0, "write results every RESULTSTEVERY steps", &adyn);
  setStringToIntegralParameter<Inpar::ALE::DivContAct>("DIVERCONT", "continue",
      "What to do if nonlinear solver does not converge?", tuple<std::string>("stop", "continue"),
      tuple<Inpar::ALE::DivContAct>(divcont_stop, divcont_continue), &adyn);

  setStringToIntegralParameter<Inpar::ALE::MeshTying>("MESHTYING", "no",
      "Flag to (de)activate mesh tying and mesh sliding algorithm",
      tuple<std::string>("no", "meshtying", "meshsliding"),
      tuple<Inpar::ALE::MeshTying>(no_meshtying, meshtying, meshsliding), &adyn);

  // Initial displacement
  setStringToIntegralParameter<Inpar::ALE::InitialDisp>("INITIALDISP", "zero_displacement",
      "Initial displacement for structure problem",
      tuple<std::string>("zero_displacement", "displacement_by_function"),
      tuple<Inpar::ALE::InitialDisp>(initdisp_zero_disp, initdisp_disp_by_function), &adyn);

  // Function to evaluate initial displacement
  Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for Initial displacement", &adyn);

  // linear solver id used for scalar ale problems
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for ale problems...", &adyn);
}



void Inpar::ALE::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // Ale update boundary condition

  Core::Conditions::ConditionDefinition linealeupdate("DESIGN ALE UPDATE LINE CONDITIONS",
      "ALEUPDATECoupling", "ALEUPDATE Coupling", Core::Conditions::ALEUPDATECoupling, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfaleupdate("DESIGN ALE UPDATE SURF CONDITIONS",
      "ALEUPDATECoupling", "ALEUPDATE Coupling", Core::Conditions::ALEUPDATECoupling, true,
      Core::Conditions::geometry_type_surface);

  const auto make_ale_update = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(selection<std::string>("COUPLING",
        {"lagrange", "heightfunction", "sphereHeightFunction", "meantangentialvelocity",
            "meantangentialvelocityscaled"},
        {.description = "", .default_value = "lagrange"}));
    cond.add_component(entry<double>("VAL"));
    cond.add_component(entry<int>("NODENORMALFUNCT"));

    condlist.emplace_back(cond);
  };

  make_ale_update(linealeupdate);
  make_ale_update(surfaleupdate);
}

FOUR_C_NAMESPACE_CLOSE

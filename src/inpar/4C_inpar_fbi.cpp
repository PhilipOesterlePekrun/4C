// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fbi.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_geometry_pair.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::FBI::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& fbi = list.sublist("FLUID BEAM INTERACTION", false, "");

  /*----------------------------------------------------------------------*/
  /* parameters for beam to fluid meshtying */

  setStringToIntegralParameter<BeamToFluidCoupling>("COUPLING", "two-way", "Type of FBI coupling",
      tuple<std::string>("two-way", "fluid", "solid"),
      tuple<BeamToFluidCoupling>(
          BeamToFluidCoupling::twoway, BeamToFluidCoupling::fluid, BeamToFluidCoupling::solid),
      &fbi);

  Core::Utils::int_parameter("STARTSTEP", 0,
      "Time Step at which to begin the fluid beam coupling. Usually this will be the first step.",
      &fbi);

  setStringToIntegralParameter<BeamToFluidPreSortStrategy>("PRESORT_STRATEGY", "bruteforce",
      "Presort strategy for the beam elements", tuple<std::string>("bruteforce", "binning"),
      tuple<BeamToFluidPreSortStrategy>(
          BeamToFluidPreSortStrategy::bruteforce, BeamToFluidPreSortStrategy::binning),
      &fbi);

  /*----------------------------------------------------------------------*/

  Teuchos::ParameterList& beam_to_fluid_meshtying =
      fbi.sublist("BEAM TO FLUID MESHTYING", false, "");

  setStringToIntegralParameter<BeamToFluidDiscretization>("MESHTYING_DISCRETIZATION", "none",
      "Type of employed meshtying discretization",
      tuple<std::string>("none", "gauss_point_to_segment", "mortar"),
      tuple<FBI::BeamToFluidDiscretization>(BeamToFluidDiscretization::none,
          BeamToFluidDiscretization::gauss_point_to_segment, BeamToFluidDiscretization::mortar),
      &beam_to_fluid_meshtying);

  setStringToIntegralParameter<BeamToFluidConstraintEnforcement>("CONSTRAINT_STRATEGY", "none",
      "Type of employed constraint enforcement strategy", tuple<std::string>("none", "penalty"),
      tuple<BeamToFluidConstraintEnforcement>(
          BeamToFluidConstraintEnforcement::none, BeamToFluidConstraintEnforcement::penalty),
      &beam_to_fluid_meshtying);

  Core::Utils::double_parameter("PENALTY_PARAMETER", 0.0,
      "Penalty parameter for beam-to-Fluid volume meshtying", &beam_to_fluid_meshtying);

  Core::Utils::double_parameter("SEARCH_RADIUS", 1000,
      "Absolute Search radius for beam-to-fluid volume meshtying. Choose carefully to not blow up "
      "memory demand but to still find all interaction pairs!",
      &beam_to_fluid_meshtying);

  setStringToIntegralParameter<Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions>(
      "MORTAR_SHAPE_FUNCTION", "none", "Shape function for the mortar Lagrange-multipliers",
      tuple<std::string>("none", "line2", "line3", "line4"),
      tuple<Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions>(
          Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::none,
          Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line2,
          Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line3,
          Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line4),
      &beam_to_fluid_meshtying);

  // Add the geometry pair input parameters.
  Inpar::GEOMETRYPAIR::set_valid_parameters_line_to3_d(beam_to_fluid_meshtying);

  /*----------------------------------------------------------------------*/

  // Create subsection for runtime output.
  Teuchos::ParameterList& beam_to_fluid_meshtying_output =
      beam_to_fluid_meshtying.sublist("RUNTIME VTK OUTPUT", false, "");

  // Whether to write visualization output at all for beam to fluid meshtying.
  Core::Utils::bool_parameter("WRITE_OUTPUT", "No",
      "Enable / disable beam-to-fluid mesh tying output.", &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("NODAL_FORCES", "No",
      "Enable / disable output of the resulting nodal forces due to beam to Fluid interaction.",
      &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("SEGMENTATION", "No",
      "Enable / disable output of segmentation points.", &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("INTEGRATION_POINTS", "No",
      "Enable / disable output of used integration points. If the meshtying method has 'forces' at "
      "the integration point, they will also be output.",
      &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("CONSTRAINT_VIOLATION", "No",
      "Enable / disable output of the constraint violation into a output_name.penalty csv file.",
      &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("MORTAR_LAMBDA_DISCRET", "No",
      "Enable / disable output of the discrete Lagrange multipliers at the node of the Lagrange "
      "multiplier shape functions.",
      &beam_to_fluid_meshtying_output);

  Core::Utils::bool_parameter("MORTAR_LAMBDA_CONTINUOUS", "No",
      "Enable / disable output of the continuous Lagrange multipliers function along the beam.",
      &beam_to_fluid_meshtying_output);

  Core::Utils::int_parameter("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS", 5,
      "Number of segments for continuous mortar output", &beam_to_fluid_meshtying_output);
}

void Inpar::FBI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{ /*-------------------------------------------------------------------*/
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_beam_to_solid.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_geometry_pair.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void Inpar::BeamToSolid::beam_to_solid_interaction_get_string(
    const Inpar::BeamInteraction::BeamInteractionConditions& interaction,
    std::array<std::string, 2>& condition_names)
{
  if (interaction ==
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying)
  {
    condition_names[0] = "BeamToSolidVolumeMeshtyingLine";
    condition_names[1] = "BeamToSolidVolumeMeshtyingVolume";
  }
  else if (interaction ==
           Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying)
  {
    condition_names[0] = "BeamToSolidSurfaceMeshtyingLine";
    condition_names[1] = "BeamToSolidSurfaceMeshtyingSurface";
  }
  else if (interaction ==
           Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact)
  {
    condition_names[0] = "BeamToSolidSurfaceContactLine";
    condition_names[1] = "BeamToSolidSurfaceContactSurface";
  }
  else
    FOUR_C_THROW("Got unexpected beam-to-solid interaction type.");
}

/**
 *
 */
void Inpar::BeamToSolid::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& beaminteraction = list.sublist("BEAM INTERACTION", false, "");

  // Beam to solid volume mesh tying parameters.
  Teuchos::ParameterList& beam_to_solid_volume_mestying =
      beaminteraction.sublist("BEAM TO SOLID VOLUME MESHTYING", false, "");
  {
    setStringToIntegralParameter<BeamToSolidContactDiscretization>("CONTACT_DISCRETIZATION", "none",
        "Type of employed contact discretization",
        tuple<std::string>("none", "gauss_point_to_segment", "mortar", "gauss_point_cross_section",
            "mortar_cross_section"),
        tuple<BeamToSolidContactDiscretization>(BeamToSolidContactDiscretization::none,
            BeamToSolidContactDiscretization::gauss_point_to_segment,
            BeamToSolidContactDiscretization::mortar,
            BeamToSolidContactDiscretization::gauss_point_cross_section,
            BeamToSolidContactDiscretization::mortar_cross_section),
        &beam_to_solid_volume_mestying);

    setStringToIntegralParameter<BeamToSolidConstraintEnforcement>("CONSTRAINT_STRATEGY", "none",
        "Type of employed constraint enforcement strategy", tuple<std::string>("none", "penalty"),
        tuple<BeamToSolidConstraintEnforcement>(
            BeamToSolidConstraintEnforcement::none, BeamToSolidConstraintEnforcement::penalty),
        &beam_to_solid_volume_mestying);

    setStringToIntegralParameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION", "none",
        "Shape function for the mortar Lagrange-multipliers",
        tuple<std::string>("none", "line2", "line3", "line4"),
        tuple<BeamToSolidMortarShapefunctions>(BeamToSolidMortarShapefunctions::none,
            BeamToSolidMortarShapefunctions::line2, BeamToSolidMortarShapefunctions::line3,
            BeamToSolidMortarShapefunctions::line4),
        &beam_to_solid_volume_mestying);

    Core::Utils::int_parameter("MORTAR_FOURIER_MODES", -1,
        "Number of fourier modes to be used for cross-section mortar coupling",
        &beam_to_solid_volume_mestying);

    Core::Utils::double_parameter("PENALTY_PARAMETER", 0.0,
        "Penalty parameter for beam-to-solid volume meshtying", &beam_to_solid_volume_mestying);

    // Add the geometry pair input parameters.
    Inpar::GEOMETRYPAIR::set_valid_parameters_line_to3_d(beam_to_solid_volume_mestying);

    // This option only has an effect during a restart simulation.
    // - No:  (default) The coupling is treated the same way as during a non restart simulation,
    //        i.e. the initial configurations (zero displacement) of the beams and solids are
    //        coupled.
    // - Yes: The beam and solid states at the restart configuration are coupled. This allows to
    //        pre-deform the structures and then couple them.
    Core::Utils::bool_parameter("COUPLE_RESTART_STATE", "No",
        "Enable / disable the coupling of the restart configuration.",
        &beam_to_solid_volume_mestying);

    setStringToIntegralParameter<BeamToSolidRotationCoupling>("ROTATION_COUPLING", "none",
        "Type of rotational coupling",
        tuple<std::string>("none", "deformation_gradient_3d_general_in_cross_section_plane",
            "polar_decomposition_2d", "deformation_gradient_y_2d", "deformation_gradient_z_2d",
            "deformation_gradient_average_2d", "fix_triad_2d", "deformation_gradient_3d_local_1",
            "deformation_gradient_3d_local_2", "deformation_gradient_3d_local_3",
            "deformation_gradient_3d_general",

            "deformation_gradient_3d_base_1"),
        tuple<BeamToSolidRotationCoupling>(BeamToSolidRotationCoupling::none,
            BeamToSolidRotationCoupling::deformation_gradient_3d_general_in_cross_section_plane,
            BeamToSolidRotationCoupling::polar_decomposition_2d,
            BeamToSolidRotationCoupling::deformation_gradient_y_2d,
            BeamToSolidRotationCoupling::deformation_gradient_z_2d,
            BeamToSolidRotationCoupling::deformation_gradient_average_2d,
            BeamToSolidRotationCoupling::fix_triad_2d,
            BeamToSolidRotationCoupling::deformation_gradient_3d_local_1,
            BeamToSolidRotationCoupling::deformation_gradient_3d_local_2,
            BeamToSolidRotationCoupling::deformation_gradient_3d_local_3,
            BeamToSolidRotationCoupling::deformation_gradient_3d_general,
            BeamToSolidRotationCoupling::deformation_gradient_3d_base_1),
        &beam_to_solid_volume_mestying);

    setStringToIntegralParameter<BeamToSolidMortarShapefunctions>(
        "ROTATION_COUPLING_MORTAR_SHAPE_FUNCTION", "none",
        "Shape function for the mortar Lagrange-multipliers",
        tuple<std::string>("none", "line2", "line3", "line4"),
        tuple<BeamToSolidMortarShapefunctions>(BeamToSolidMortarShapefunctions::none,
            BeamToSolidMortarShapefunctions::line2, BeamToSolidMortarShapefunctions::line3,
            BeamToSolidMortarShapefunctions::line4),
        &beam_to_solid_volume_mestying);

    Core::Utils::double_parameter("ROTATION_COUPLING_PENALTY_PARAMETER", 0.0,
        "Penalty parameter for rotational coupling in beam-to-solid volume mesh tying",
        &beam_to_solid_volume_mestying);
  }

  // Beam to solid volume mesh tying output parameters.
  Teuchos::ParameterList& beam_to_solid_volume_mestying_output =
      beam_to_solid_volume_mestying.sublist("RUNTIME VTK OUTPUT", false, "");
  {
    // Whether to write visualization output at all for btsvmt.
    Core::Utils::bool_parameter("WRITE_OUTPUT", "No",
        "Enable / disable beam-to-solid volume mesh tying output.",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("NODAL_FORCES", "No",
        "Enable / disable output of the resulting nodal forces due to beam to solid interaction.",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("MORTAR_LAMBDA_DISCRET", "No",
        "Enable / disable output of the discrete Lagrange multipliers at the node of the Lagrange "
        "multiplier shape functions.",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("MORTAR_LAMBDA_CONTINUOUS", "No",
        "Enable / disable output of the continuous Lagrange multipliers function along the beam.",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::int_parameter("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS", 5,
        "Number of segments for continuous mortar output", &beam_to_solid_volume_mestying_output);

    Core::Utils::int_parameter("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS_CIRCUMFERENCE", 8,
        "Number of segments for continuous mortar output along the beam cross-section "
        "circumference",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("SEGMENTATION", "No",
        "Enable / disable output of segmentation points.", &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("INTEGRATION_POINTS", "No",
        "Enable / disable output of used integration points. If the contact method has 'forces' at "
        "the integration point, they will also be output.",
        &beam_to_solid_volume_mestying_output);

    Core::Utils::bool_parameter("UNIQUE_IDS", "No",
        "Enable / disable output of unique IDs (mainly for testing of created VTK files).",
        &beam_to_solid_volume_mestying_output);
  }

  // Beam to solid surface mesh tying parameters.
  Teuchos::ParameterList& beam_to_solid_surface_mestying =
      beaminteraction.sublist("BEAM TO SOLID SURFACE MESHTYING", false, "");
  {
    setStringToIntegralParameter<BeamToSolidContactDiscretization>("CONTACT_DISCRETIZATION", "none",
        "Type of employed contact discretization",
        tuple<std::string>("none", "gauss_point_to_segment", "mortar"),
        tuple<BeamToSolidContactDiscretization>(BeamToSolidContactDiscretization::none,
            BeamToSolidContactDiscretization::gauss_point_to_segment,
            BeamToSolidContactDiscretization::mortar),
        &beam_to_solid_surface_mestying);

    setStringToIntegralParameter<BeamToSolidConstraintEnforcement>("CONSTRAINT_STRATEGY", "none",
        "Type of employed constraint enforcement strategy", tuple<std::string>("none", "penalty"),
        tuple<BeamToSolidConstraintEnforcement>(
            BeamToSolidConstraintEnforcement::none, BeamToSolidConstraintEnforcement::penalty),
        &beam_to_solid_surface_mestying);

    setStringToIntegralParameter<BeamToSolidSurfaceCoupling>("COUPLING_TYPE", "none",
        "How the coupling constraints are formulated/",
        tuple<std::string>("none", "reference_configuration_forced_to_zero",
            "reference_configuration_forced_to_zero_fad", "displacement", "displacement_fad",
            "consistent_fad"),
        tuple<BeamToSolidSurfaceCoupling>(BeamToSolidSurfaceCoupling::none,
            BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero,
            BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero_fad,
            BeamToSolidSurfaceCoupling::displacement, BeamToSolidSurfaceCoupling::displacement_fad,
            BeamToSolidSurfaceCoupling::consistent_fad),
        &beam_to_solid_surface_mestying);

    setStringToIntegralParameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION", "none",
        "Shape function for the mortar Lagrange-multipliers",
        tuple<std::string>("none", "line2", "line3", "line4"),
        tuple<BeamToSolidMortarShapefunctions>(BeamToSolidMortarShapefunctions::none,
            BeamToSolidMortarShapefunctions::line2, BeamToSolidMortarShapefunctions::line3,
            BeamToSolidMortarShapefunctions::line4),
        &beam_to_solid_surface_mestying);

    Core::Utils::double_parameter("PENALTY_PARAMETER", 0.0,
        "Penalty parameter for beam-to-solid surface meshtying", &beam_to_solid_surface_mestying);

    // Parameters for rotational coupling.
    Core::Utils::bool_parameter("ROTATIONAL_COUPLING", "No", "Enable / disable rotational coupling",
        &beam_to_solid_surface_mestying);
    Core::Utils::double_parameter("ROTATIONAL_COUPLING_PENALTY_PARAMETER", 0.0,
        "Penalty parameter for beam-to-solid surface rotational meshtying",
        &beam_to_solid_surface_mestying);
    setStringToIntegralParameter<BeamToSolidSurfaceRotationCoupling>(
        "ROTATIONAL_COUPLING_SURFACE_TRIAD", "none", "Construction method for surface triad",
        tuple<std::string>("none", "surface_cross_section_director", "averaged"),
        tuple<BeamToSolidSurfaceRotationCoupling>(BeamToSolidSurfaceRotationCoupling::none,
            BeamToSolidSurfaceRotationCoupling::surface_cross_section_director,
            BeamToSolidSurfaceRotationCoupling::averaged),
        &beam_to_solid_surface_mestying);

    // Add the geometry pair input parameters.
    Inpar::GEOMETRYPAIR::set_valid_parameters_line_to3_d(beam_to_solid_surface_mestying);

    // Add the surface options.
    Inpar::GEOMETRYPAIR::set_valid_parameters_line_to_surface(beam_to_solid_surface_mestying);
  }

  // Beam to solid surface contact parameters.
  Teuchos::ParameterList& beam_to_solid_surface_contact =
      beaminteraction.sublist("BEAM TO SOLID SURFACE CONTACT", false, "");
  {
    setStringToIntegralParameter<BeamToSolidContactDiscretization>("CONTACT_DISCRETIZATION", "none",
        "Type of employed contact discretization",
        tuple<std::string>("none", "gauss_point_to_segment", "mortar"),
        tuple<BeamToSolidContactDiscretization>(BeamToSolidContactDiscretization::none,
            BeamToSolidContactDiscretization::gauss_point_to_segment,
            BeamToSolidContactDiscretization::mortar),
        &beam_to_solid_surface_contact);

    setStringToIntegralParameter<BeamToSolidConstraintEnforcement>("CONSTRAINT_STRATEGY", "none",
        "Type of employed constraint enforcement strategy", tuple<std::string>("none", "penalty"),
        tuple<BeamToSolidConstraintEnforcement>(
            BeamToSolidConstraintEnforcement::none, BeamToSolidConstraintEnforcement::penalty),
        &beam_to_solid_surface_contact);

    Core::Utils::double_parameter("PENALTY_PARAMETER", 0.0,
        "Penalty parameter for beam-to-solid surface contact", &beam_to_solid_surface_contact);

    setStringToIntegralParameter<BeamToSolidSurfaceContact>("CONTACT_TYPE", "none",
        "How the contact constraints are formulated",
        tuple<std::string>("none", "gap_variation", "potential"),
        tuple<BeamToSolidSurfaceContact>(BeamToSolidSurfaceContact::none,
            BeamToSolidSurfaceContact::gap_variation, BeamToSolidSurfaceContact::potential),
        &beam_to_solid_surface_contact);

    setStringToIntegralParameter<BeamToSolidSurfaceContactPenaltyLaw>("PENALTY_LAW", "none",
        "Type of penalty law", tuple<std::string>("none", "linear", "linear_quadratic"),
        tuple<BeamToSolidSurfaceContactPenaltyLaw>(BeamToSolidSurfaceContactPenaltyLaw::none,
            BeamToSolidSurfaceContactPenaltyLaw::linear,
            BeamToSolidSurfaceContactPenaltyLaw::linear_quadratic),
        &beam_to_solid_surface_contact);

    Core::Utils::double_parameter("PENALTY_PARAMETER_G0", 0.0,
        "First penalty regularization parameter G0 >=0: For gap<G0 contact is active",
        &beam_to_solid_surface_contact);

    setStringToIntegralParameter<BeamToSolidSurfaceContactMortarDefinedIn>(
        "MORTAR_CONTACT_DEFINED_IN", "none", "Configuration where the mortar contact is defined",
        tuple<std::string>("none", "reference_configuration", "current_configuration"),
        tuple<BeamToSolidSurfaceContactMortarDefinedIn>(
            BeamToSolidSurfaceContactMortarDefinedIn::none,
            BeamToSolidSurfaceContactMortarDefinedIn::reference_configuration,
            BeamToSolidSurfaceContactMortarDefinedIn::current_configuration),
        &beam_to_solid_surface_contact);

    // Add the geometry pair input parameters.
    Inpar::GEOMETRYPAIR::set_valid_parameters_line_to3_d(beam_to_solid_surface_contact);

    // Add the surface options.
    Inpar::GEOMETRYPAIR::set_valid_parameters_line_to_surface(beam_to_solid_surface_contact);

    // Define the mortar shape functions for contact
    setStringToIntegralParameter<BeamToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION", "none",
        "Shape function for the mortar Lagrange-multipliers", tuple<std::string>("none", "line2"),
        tuple<BeamToSolidMortarShapefunctions>(
            BeamToSolidMortarShapefunctions::none, BeamToSolidMortarShapefunctions::line2),
        &beam_to_solid_surface_contact);
  }

  // Beam to solid surface parameters.
  Teuchos::ParameterList& beam_to_solid_surface =
      beaminteraction.sublist("BEAM TO SOLID SURFACE", false, "");

  // Beam to solid surface output parameters.
  Teuchos::ParameterList& beam_to_solid_surface_output =
      beam_to_solid_surface.sublist("RUNTIME VTK OUTPUT", false, "");
  {
    // Whether to write visualization output at all.
    Core::Utils::bool_parameter("WRITE_OUTPUT", "No",
        "Enable / disable beam-to-solid volume mesh tying output.", &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("NODAL_FORCES", "No",
        "Enable / disable output of the resulting nodal forces due to beam to solid interaction.",
        &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("AVERAGED_NORMALS", "No",
        "Enable / disable output of averaged nodal normals on the surface.",
        &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("MORTAR_LAMBDA_DISCRET", "No",
        "Enable / disable output of the discrete Lagrange multipliers at the node of the Lagrange "
        "multiplier shape functions.",
        &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("MORTAR_LAMBDA_CONTINUOUS", "No",
        "Enable / disable output of the continuous Lagrange multipliers function along the beam.",
        &beam_to_solid_surface_output);

    Core::Utils::int_parameter("MORTAR_LAMBDA_CONTINUOUS_SEGMENTS", 5,
        "Number of segments for continuous mortar output", &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("SEGMENTATION", "No",
        "Enable / disable output of segmentation points.", &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("INTEGRATION_POINTS", "No",
        "Enable / disable output of used integration points. If the contact method has 'forces' at "
        "the integration point, they will also be output.",
        &beam_to_solid_surface_output);

    Core::Utils::bool_parameter("UNIQUE_IDS", "No",
        "Enable / disable output of unique IDs (mainly for testing of created VTK files).",
        &beam_to_solid_surface_output);
  }
}

/**
 *
 */
void Inpar::BeamToSolid::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  // Beam-to-volume mesh tying conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_volume_meshtying,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_volume_meshtying_condition(
        "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME", condition_names[1],
        "Beam-to-volume mesh tying conditions - volume definition",
        Core::Conditions::BeamToSolidVolumeMeshtyingVolume, true,
        Core::Conditions::geometry_type_volume);
    beam_to_solid_volume_meshtying_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_volume_meshtying_condition);

    beam_to_solid_volume_meshtying_condition = Core::Conditions::ConditionDefinition(
        "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE", condition_names[0],
        "Beam-to-volume mesh tying conditions - line definition",
        Core::Conditions::BeamToSolidVolumeMeshtyingLine, true,
        Core::Conditions::geometry_type_line);
    beam_to_solid_volume_meshtying_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_volume_meshtying_condition);
  }

  // Beam-to-surface mesh tying conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_meshtying,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_surface_meshtying_condition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE", condition_names[1],
        "Beam-to-surface mesh tying conditions - surface definition",
        Core::Conditions::BeamToSolidSurfaceMeshtyingSurface, true,
        Core::Conditions::geometry_type_surface);
    beam_to_solid_surface_meshtying_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_meshtying_condition);

    beam_to_solid_surface_meshtying_condition = Core::Conditions::ConditionDefinition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE", condition_names[0],
        "Beam-to-surface mesh tying conditions - line definition",
        Core::Conditions::BeamToSolidSurfaceMeshtyingLine, true,
        Core::Conditions::geometry_type_line);
    beam_to_solid_surface_meshtying_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_meshtying_condition);
  }

  // Beam-to-surface contact conditions.
  {
    std::array<std::string, 2> condition_names;
    beam_to_solid_interaction_get_string(
        Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact,
        condition_names);

    Core::Conditions::ConditionDefinition beam_to_solid_surface_contact_condition(
        "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE", condition_names[1],
        "Beam-to-surface contact conditions - surface definition",
        Core::Conditions::BeamToSolidSurfaceContactSurface, true,
        Core::Conditions::geometry_type_surface);
    beam_to_solid_surface_contact_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_contact_condition);

    beam_to_solid_surface_contact_condition =
        Core::Conditions::ConditionDefinition("BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
            condition_names[0], "Beam-to-surface contact conditions - line definition",
            Core::Conditions::BeamToSolidSurfaceContactLine, true,
            Core::Conditions::geometry_type_line);
    beam_to_solid_surface_contact_condition.add_component(entry<int>("COUPLING_ID"));
    condlist.push_back(beam_to_solid_surface_contact_condition);
  }
}

FOUR_C_NAMESPACE_CLOSE

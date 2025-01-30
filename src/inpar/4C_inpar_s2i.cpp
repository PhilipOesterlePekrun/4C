// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_s2i.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------*
 | set valid parameters for scatra-scatra interface coupling   fang 01/16 |
 *------------------------------------------------------------------------*/
void Inpar::S2I::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& s2icoupling =
      list.sublist("SCALAR TRANSPORT DYNAMIC", true)
          .sublist(
              "S2I COUPLING", false, "control parameters for scatra-scatra interface coupling");

  // type of mortar meshtying
  setStringToIntegralParameter<CouplingType>("COUPLINGTYPE", "Undefined",
      "type of mortar meshtying",
      tuple<std::string>("Undefined", "MatchingNodes", "StandardMortar", "SaddlePointMortar_Petrov",
          "SaddlePointMortar_Bubnov", "CondensedMortar_Petrov", "CondensedMortar_Bubnov",
          "StandardNodeToSegment"),
      tuple<CouplingType>(coupling_undefined, coupling_matching_nodes, coupling_mortar_standard,
          coupling_mortar_saddlepoint_petrov, coupling_mortar_saddlepoint_bubnov,
          coupling_mortar_condensed_petrov, coupling_mortar_condensed_bubnov,
          coupling_nts_standard),
      &s2icoupling);

  // flag for interface side underlying Lagrange multiplier definition
  setStringToIntegralParameter<InterfaceSides>("LMSIDE", "slave",
      "flag for interface side underlying Lagrange multiplier definition",
      tuple<std::string>("slave", "master"), tuple<InterfaceSides>(side_slave, side_master),
      &s2icoupling);

  // flag for evaluation of interface linearizations and residuals on slave side only
  Core::Utils::bool_parameter("SLAVEONLY", "No",
      "flag for evaluation of interface linearizations and residuals on slave side only",
      &s2icoupling);

  // node-to-segment projection tolerance
  Core::Utils::double_parameter(
      "NTSPROJTOL", 0.0, "node-to-segment projection tolerance", &s2icoupling);

  // flag for evaluation of scatra-scatra interface coupling involving interface layer growth
  setStringToIntegralParameter<GrowthEvaluation>("INTLAYERGROWTH_EVALUATION", "none",
      "flag for evaluation of scatra-scatra interface coupling involving interface layer growth",
      tuple<std::string>("none", "monolithic", "semi-implicit"),
      tuple<GrowthEvaluation>(
          growth_evaluation_none, growth_evaluation_monolithic, growth_evaluation_semi_implicit),
      &s2icoupling);

  // local Newton-Raphson convergence tolerance for scatra-scatra interface coupling involving
  // interface layer growth
  Core::Utils::double_parameter("INTLAYERGROWTH_CONVTOL", 1.e-12,
      "local Newton-Raphson convergence tolerance for scatra-scatra interface coupling involving "
      "interface layer growth",
      &s2icoupling);

  // maximum number of local Newton-Raphson iterations for scatra-scatra interface coupling
  // involving interface layer growth
  Core::Utils::int_parameter("INTLAYERGROWTH_ITEMAX", 5,
      "maximum number of local Newton-Raphson iterations for scatra-scatra interface coupling "
      "involving interface layer growth",
      &s2icoupling);

  // ID of linear solver for monolithic scatra-scatra interface coupling involving interface layer
  // growth
  Core::Utils::int_parameter("INTLAYERGROWTH_LINEAR_SOLVER", -1,
      "ID of linear solver for monolithic scatra-scatra interface coupling involving interface "
      "layer growth",
      &s2icoupling);

  // modified time step size for scatra-scatra interface coupling involving interface layer growth
  Core::Utils::double_parameter("INTLAYERGROWTH_TIMESTEP", -1.,
      "modified time step size for scatra-scatra interface coupling involving interface layer "
      "growth",
      &s2icoupling);

  Core::Utils::bool_parameter("MESHTYING_CONDITIONS_INDEPENDENT_SETUP", "No",
      "mesh tying for different conditions should be setup independently", &s2icoupling);

  Core::Utils::bool_parameter("OUTPUT_INTERFACE_FLUX", "No",
      "evaluate integral of coupling flux on slave side for each s2i condition and write it to csv "
      "file",
      &s2icoupling);
}


/*------------------------------------------------------------------------*
 | set valid conditions for scatra-scatra interface coupling   fang 01/16 |
 *------------------------------------------------------------------------*/
void Inpar::S2I::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface mesh tying condition
  {
    // definition of scatra-scatra interface mesh tying line condition
    Core::Conditions::ConditionDefinition s2imeshtyingline("DESIGN S2I MESHTYING LINE CONDITIONS",
        "S2IMeshtying", "Scatra-scatra line interface mesh tying", Core::Conditions::S2IMeshtying,
        true, Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface mesh tying surface condition
    Core::Conditions::ConditionDefinition s2imeshtyingsurf("DESIGN S2I MESHTYING SURF CONDITIONS",
        "S2IMeshtying", "Scatra-scatra surface interface mesh tying",
        Core::Conditions::S2IMeshtying, true, Core::Conditions::geometry_type_surface);

    const auto make_s2imeshtying = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(entry<int>("ConditionID"));
      cond.add_component(selection<int>("INTERFACE_SIDE",
          {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
              {"Master", Inpar::S2I::side_master}},
          {.description = "interface side"}));
      cond.add_component(entry<int>("S2I_KINETICS_ID"));

      condlist.push_back(cond);
    };

    make_s2imeshtying(s2imeshtyingline);
    make_s2imeshtying(s2imeshtyingsurf);
  }

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface kinetics condition
  {
    // definition of scatra-scatra interface kinetics point condition
    Core::Conditions::ConditionDefinition s2ikineticspoint("DESIGN S2I KINETICS POINT CONDITIONS",
        "S2IKinetics", "Scatra-scatra line interface kinetics", Core::Conditions::S2IKinetics, true,
        Core::Conditions::geometry_type_point);

    // definition of scatra-scatra interface kinetics line condition
    Core::Conditions::ConditionDefinition s2ikineticsline("DESIGN S2I KINETICS LINE CONDITIONS",
        "S2IKinetics", "Scatra-scatra line interface kinetics", Core::Conditions::S2IKinetics, true,
        Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface kinetics surface condition
    Core::Conditions::ConditionDefinition s2ikineticssurf("DESIGN S2I KINETICS SURF CONDITIONS",
        "S2IKinetics", "Scatra-scatra surface interface kinetics", Core::Conditions::S2IKinetics,
        true, Core::Conditions::geometry_type_surface);

    // Macro-micro coupling condition for micro scale in multi-scale scalar transport problems
    Core::Conditions::ConditionDefinition multiscalecouplingpoint(
        "DESIGN SCATRA MULTI-SCALE COUPLING POINT CONDITIONS", "ScatraMultiScaleCoupling",
        "Scalar transport multi-scale coupling condition",
        Core::Conditions::ScatraMultiScaleCoupling, false, Core::Conditions::geometry_type_point);

    std::vector<Core::IO::InputSpec> kinetic_model_choices;
    {
      {
        // constant and linear permeability
        auto constlinperm = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"ConstantPermeability", Inpar::S2I::kinetics_constperm},
                    {"LinearPermeability", Inpar::S2I::kinetics_linearperm},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<double>>("PERMEABILITIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<bool>("IS_PSEUDO_CONTACT"),
        });
        kinetic_model_choices.emplace_back(std::move(constlinperm));
      }

      {
        auto butler_volmer = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-Volmer", Inpar::S2I::kinetics_butlervolmer},
                    {"Butler-Volmer_Linearized", Inpar::S2I::kinetics_butlervolmerlinearized},
                    {"Butler-VolmerReduced", Inpar::S2I::kinetics_butlervolmerreduced},
                    {"Butler-VolmerReduced_Linearized",
                        Inpar::S2I::kinetics_butlervolmerreducedlinearized},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
        });
        kinetic_model_choices.emplace_back(std::move(butler_volmer));
      }

      {
        auto butler_volmer_peltier = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-Volmer-Peltier", Inpar::S2I::kinetics_butlervolmerpeltier},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
            entry<double>("PELTIER"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_peltier));
      }

      {
        auto butler_volmer_reduced_capacitance = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_Capacitance",
                        Inpar::S2I::kinetics_butlervolmerreducedcapacitance},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("CAPACITANCE"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_capacitance));
      }

      {
        auto butler_volmer_resistance = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-Volmer_Resistance", Inpar::S2I::kinetics_butlervolmerresistance},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
            entry<double>("RESISTANCE"),
            entry<double>("CONVTOL_IMPLBUTLERVOLMER"),
            entry<int>("ITEMAX_IMPLBUTLERVOLMER"),
        });
        kinetic_model_choices.emplace_back(std::move(butler_volmer_resistance));
      }

      {
        auto butler_volmer_reduced_with_resistance = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_Resistance",
                        Inpar::S2I::kinetics_butlervolmerreducedresistance},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
            entry<double>("RESISTANCE"),
            entry<double>("CONVTOL_IMPLBUTLERVOLMER"),
            entry<int>("ITEMAX_IMPLBUTLERVOLMER"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_with_resistance));
      }

      {
        auto butler_volmer_reduced_thermo = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_ThermoResistance",
                        Inpar::S2I::kinetics_butlervolmerreducedthermoresistance},
                }),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
            entry<bool>("IS_PSEUDO_CONTACT"),
            entry<double>("THERMOPERM"),
            entry<double>("MOLAR_HEAT_CAPACITY"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_thermo));
      }

      {
        auto constant_interface_resistance = all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"ConstantInterfaceResistance",
                        Inpar::S2I::kinetics_constantinterfaceresistance},
                }),
            entry<std::vector<int>>("ONOFF", {.size = 2}),
            entry<double>("RESISTANCE"),
            entry<int>("E-"),
            entry<bool>("IS_PSEUDO_CONTACT"),
        });

        kinetic_model_choices.emplace_back(std::move(constant_interface_resistance));
      }

      {
        // no interface flux
        auto noflux = selection<int>(
            "KINETIC_MODEL", {
                                 {"NoInterfaceFlux", Inpar::S2I::kinetics_nointerfaceflux},
                             });

        kinetic_model_choices.emplace_back(std::move(noflux));
      }

      multiscalecouplingpoint.add_component(one_of(kinetic_model_choices));
    }

    auto interface_side_options = one_of({
        all_of({
            selection<int>("INTERFACE_SIDE", {{"Master", side_master}}),
        }),
        all_of({
            selection<int>("INTERFACE_SIDE", {{"Undefined", side_undefined}}),
        }),
        all_of({
            selection<int>("INTERFACE_SIDE", {{"Slave", side_slave}}),
            one_of(kinetic_model_choices),
        }),
    });

    const auto make_s2ikinetics = [&condlist, &interface_side_options](
                                      Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(entry<int>("ConditionID"));
      cond.add_component(interface_side_options);

      condlist.push_back(cond);
    };

    make_s2ikinetics(s2ikineticspoint);
    make_s2ikinetics(s2ikineticsline);
    make_s2ikinetics(s2ikineticssurf);


    condlist.emplace_back(multiscalecouplingpoint);
  }



  /*--------------------------------------------------------------------*/
  // scatra-scatra interface coupling involving interface layer growth
  {
    // definition of scatra-scatra interface coupling line condition involving interface layer
    // growth
    Core::Conditions::ConditionDefinition s2igrowthline(
        "DESIGN S2I KINETICS GROWTH LINE CONDITIONS", "S2IKineticsGrowth",
        "Scatra-scatra line interface layer growth kinetics", Core::Conditions::S2IKineticsGrowth,
        true, Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface coupling surface condition involving interface layer
    // growth
    Core::Conditions::ConditionDefinition s2igrowthsurf(
        "DESIGN S2I KINETICS GROWTH SURF CONDITIONS", "S2IKineticsGrowth",
        "Scatra-scatra surface interface layer growth kinetics",
        Core::Conditions::S2IKineticsGrowth, true, Core::Conditions::geometry_type_surface);

    auto butler_volmer = all_of({
        entry<int>("NUMSCAL"),
        entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
        entry<int>("E-"),
        entry<double>("K_R"),
        entry<double>("ALPHA_A"),
        entry<double>("ALPHA_C"),
        entry<double>("MOLMASS"),
        entry<double>("DENSITY"),
        entry<double>("CONDUCTIVITY"),
        selection<int>("REGTYPE",
            {
                {"none", Inpar::S2I::regularization_none},
                {"polynomial", Inpar::S2I::regularization_polynomial},
                {"Hein", Inpar::S2I::regularization_hein},
                {"trigonometrical", Inpar::S2I::regularization_trigonometrical},
            }),
        entry<double>("REGPAR"),
        entry<double>("INITTHICKNESS"),
    });

    const auto make_s2igrowth = [&condlist, &butler_volmer](
                                    Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(entry<int>("ConditionID"));
      cond.add_component(
          selection<int>("KINETIC_MODEL", {{"Butler-Volmer", growth_kinetics_butlervolmer}}));
      cond.add_component(butler_volmer);

      condlist.emplace_back(cond);
    };

    make_s2igrowth(s2igrowthline);
    make_s2igrowth(s2igrowthsurf);
  }

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface with micro-macro coupling for space-charge layers
  {
    Core::Conditions::ConditionDefinition s2isclcond("DESIGN S2I SCL COUPLING SURF CONDITIONS",
        "S2ISCLCoupling", "Scatra-scatra surface with SCL micro-macro coupling between",
        Core::Conditions::S2ISCLCoupling, true, Core::Conditions::geometry_type_surface);

    s2isclcond.add_component(selection<int>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface side"}));

    condlist.emplace_back(s2isclcond);
  }
}

FOUR_C_NAMESPACE_CLOSE

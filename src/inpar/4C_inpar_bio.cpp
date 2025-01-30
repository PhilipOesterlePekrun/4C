// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_bio.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::ArtDyn::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;
  Teuchos::ParameterList& andyn = list.sublist("ARTERIAL DYNAMIC", false, "");

  setStringToIntegralParameter<TimeIntegrationScheme>("DYNAMICTYPE", "ExpTaylorGalerkin",
      "Explicit Taylor Galerkin Scheme", tuple<std::string>("ExpTaylorGalerkin", "Stationary"),
      tuple<TimeIntegrationScheme>(tay_gal, stationary), &andyn);

  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", &andyn);
  Core::Utils::int_parameter("NUMSTEP", 0, "Number of Time Steps", &andyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "total simulation time", &andyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", &andyn);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", &andyn);

  Core::Utils::bool_parameter(
      "SOLVESCATRA", "no", "Flag to (de)activate solving scalar transport in blood", &andyn);

  // number of linear solver used for arterial dynamics
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for arterial dynamics", &andyn);

  // initial function number
  Core::Utils::int_parameter("INITFUNCNO", -1, "function number for artery initial field", &andyn);

  // type of initial field
  setStringToIntegralParameter<InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for artery problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<InitialField>(
          initfield_zero_field, initfield_field_by_function, initfield_field_by_condition),
      &andyn);
}



void Inpar::ArteryNetwork::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& redtisdyn =
      list.sublist("COUPLED REDUCED-D AIRWAYS AND TISSUE DYNAMIC", false, "");
  Core::Utils::double_parameter("CONVTOL_P", 1E-6,
      "Coupled red_airway and tissue iteration convergence for pressure", &redtisdyn);
  Core::Utils::double_parameter("CONVTOL_Q", 1E-6,
      "Coupled red_airway and tissue iteration convergence for flux", &redtisdyn);
  Core::Utils::int_parameter("MAXITER", 5, "Maximum coupling iterations", &redtisdyn);
  setStringToIntegralParameter<Relaxtype3D0D>("RELAXTYPE", "norelaxation",
      "Dynamic Relaxation Type",
      tuple<std::string>("norelaxation", "fixedrelaxation", "Aitken", "SD"),
      tuple<Relaxtype3D0D>(norelaxation, fixedrelaxation, Aitken, SD), &redtisdyn);
  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", &redtisdyn);
  Core::Utils::int_parameter("NUMSTEP", 1, "Number of Time Steps", &redtisdyn);
  Core::Utils::double_parameter("MAXTIME", 4.0, "", &redtisdyn);
  Core::Utils::double_parameter("NORMAL", 1.0, "", &redtisdyn);
}



void Inpar::ArteryNetwork::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // 1D-Artery connector condition

  Core::Conditions::ConditionDefinition art_connection_bc(
      "DESIGN NODE 1D ARTERY JUNCTION CONDITIONS", "ArtJunctionCond",
      "Artery junction boundary condition", Core::Conditions::ArtJunctionCond, true,
      Core::Conditions::geometry_type_point);

  art_connection_bc.add_component(entry<int>("ConditionID"));
  art_connection_bc.add_component(entry<double>("Kr"));

  condlist.push_back(art_connection_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery prescribed BC

  Core::Conditions::ConditionDefinition art_in_bc("DESIGN NODE 1D ARTERY PRESCRIBED CONDITIONS",
      "ArtPrescribedCond", "Artery prescribed boundary condition",
      Core::Conditions::ArtPrescribedCond, true, Core::Conditions::geometry_type_point);
  art_in_bc.add_component(selection<std::string>("boundarycond",
      {"flow", "pressure", "velocity", "area", "characteristicWave"},
      {.description = "boundarycond", .default_value = "flow"}));
  art_in_bc.add_component(selection<std::string>(
      "type", {"forced", "absorbing"}, {.description = "type", .default_value = "forced"}));

  art_in_bc.add_component(entry<std::vector<double>>("VAL", {.description = "values", .size = 2}));
  art_in_bc.add_component(
      entry<std::vector<Noneable<int>>>("curve", {.description = "curve ids", .size = 2}));

  condlist.push_back(art_in_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery reflective BC
  Core::Conditions::ConditionDefinition art_rf_bc("DESIGN NODE 1D ARTERY REFLECTIVE CONDITIONS",
      "ArtRfCond", "Artery reflection condition", Core::Conditions::ArtRfCond, true,
      Core::Conditions::geometry_type_point);

  art_rf_bc.add_component(entry<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  art_rf_bc.add_component(
      entry<std::vector<Noneable<int>>>("curve", {.description = "curve", .size = 1}));
  condlist.push_back(art_rf_bc);


  /*--------------------------------------------------------------------*/
  // 1D artery in/out condition

  Core::Conditions::ConditionDefinition art_in_outlet_bc(
      "DESIGN NODE 1D ARTERY IN_OUTLET CONDITIONS", "ArtInOutCond",
      "Artery terminal in_outlet condition", Core::Conditions::ArtInOutletCond, true,
      Core::Conditions::geometry_type_point);

  art_in_outlet_bc.add_component(selection<std::string>("terminaltype", {"inlet", "outlet"},
      {.description = "terminaltype", .default_value = "inlet"}));

  condlist.push_back(art_in_outlet_bc);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC

  Core::Conditions::ConditionDefinition artcoup(
      "DESIGN NODE 1D ARTERY TO POROFLUID COUPLING CONDITIONS", "ArtPorofluidCouplConNodebased",
      "Artery coupling with porofluid", Core::Conditions::ArtPorofluidCouplingCondNodebased, true,
      Core::Conditions::geometry_type_point);

  artcoup.add_component(entry<int>("COUPID"));

  condlist.push_back(artcoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC

  Core::Conditions::ConditionDefinition artscatracoup(
      "DESIGN NODE 1D ARTERY TO SCATRA COUPLING CONDITIONS", "ArtScatraCouplConNodebased",
      "Artery coupling with porofluid", Core::Conditions::ArtScatraCouplingCondNodebased, true,
      Core::Conditions::geometry_type_point);

  artscatracoup.add_component(entry<int>("COUPID"));

  condlist.push_back(artscatracoup);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-porofluid coupling BC Node-To-Point
  Core::Conditions::ConditionDefinition artcoup_ntp(
      "DESIGN 1D ARTERY/AIRWAY TO POROFLUID NONCONF COUPLING CONDITIONS",
      "ArtPorofluidCouplConNodeToPoint", "Artery coupling with porofluid nonconf",
      Core::Conditions::ArtPorofluidCouplingCondNodeToPoint, true,
      Core::Conditions::geometry_type_point);

  artcoup_ntp.add_component(selection<std::string>("COUPLING_TYPE", {"ARTERY", "AIRWAY"},
      {.description = "coupling type", .default_value = "ARTERY"}));
  artcoup_ntp.add_component(entry<int>("NUMDOF"));
  artcoup_ntp.add_component(entry<std::vector<int>>(
      "COUPLEDDOF_REDUCED", {.description = "coupling dofs of reduced airways or arteries",
                                .size = from_parameter<int>("NUMDOF")}));
  artcoup_ntp.add_component(entry<std::vector<int>>("COUPLEDDOF_PORO",
      {.description = "coupling dofs in porous domain", .size = from_parameter<int>("NUMDOF")}));
  artcoup_ntp.add_component(entry<double>("PENALTY"));

  condlist.push_back(artcoup_ntp);

  /*--------------------------------------------------------------------*/
  // 1D artery-to-scatra coupling BC Node-To-Point
  Core::Conditions::ConditionDefinition artscatracoup_ntp(
      "DESIGN 1D ARTERY/AIRWAY TO SCATRA NONCONF COUPLING CONDITIONS",
      "ArtScatraCouplConNodeToPoint", "Artery coupling with scatra nonconf",
      Core::Conditions::ArtScatraCouplingCondNodeToPoint, true,
      Core::Conditions::geometry_type_point);

  artscatracoup_ntp.add_component(selection<std::string>("COUPLING_TYPE", {"ARTERY", "AIRWAY"},
      {.description = "coupling type", .default_value = "ARTERY"}));
  artscatracoup_ntp.add_component(entry<int>("NUMDOF"));
  artscatracoup_ntp.add_component(entry<std::vector<int>>(
      "COUPLEDDOF_REDUCED", {.description = "coupling dofs of reduced airways or arteries",
                                .size = from_parameter<int>("NUMDOF")}));
  artscatracoup_ntp.add_component(entry<std::vector<int>>("COUPLEDDOF_PORO",
      {.description = "coupling dofs in porous domain", .size = from_parameter<int>("NUMDOF")}));
  artscatracoup_ntp.add_component(entry<double>("PENALTY"));

  condlist.push_back(artscatracoup_ntp);
}



void Inpar::BioFilm::set_valid_parameters(Teuchos::ParameterList& list)
{
  Teuchos::ParameterList& biofilmcontrol =
      list.sublist("BIOFILM CONTROL", false, "control parameters for biofilm problems\n");

  Core::Utils::bool_parameter(
      "BIOFILMGROWTH", "No", "Scatra algorithm for biofilm growth", &biofilmcontrol);
  Core::Utils::bool_parameter("AVGROWTH", "No",
      "The calculation of growth parameters is based on averaged values", &biofilmcontrol);
  Core::Utils::double_parameter(
      "FLUXCOEF", 0.0, "Coefficient for growth due to scalar flux", &biofilmcontrol);
  Core::Utils::double_parameter("NORMFORCEPOSCOEF", 0.0,
      "Coefficient for erosion due to traction normal surface forces", &biofilmcontrol);
  Core::Utils::double_parameter("NORMFORCENEGCOEF", 0.0,
      "Coefficient for erosion due to compression normal surface forces", &biofilmcontrol);
  Core::Utils::double_parameter("TANGONEFORCECOEF", 0.0,
      "Coefficient for erosion due to the first tangential surface force", &biofilmcontrol);
  Core::Utils::double_parameter("TANGTWOFORCECOEF", 0.0,
      "Coefficient for erosion due to the second tangential surface force", &biofilmcontrol);
  Core::Utils::double_parameter(
      "BIOTIMESTEP", 0.05, "Time step size for biofilm growth", &biofilmcontrol);
  Core::Utils::int_parameter(
      "BIONUMSTEP", 0, "Maximum number of steps for biofilm growth", &biofilmcontrol);
  Core::Utils::bool_parameter(
      "OUTPUT_GMSH", "No", "Do you want to write Gmsh postprocessing files?", &biofilmcontrol);
}



void Inpar::BioFilm::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // Additional coupling for biofilm growth
  Core::Conditions::ConditionDefinition surfbiogr("DESIGN BIOFILM GROWTH COUPLING SURF CONDITIONS",
      "BioGrCoupling", "BioGrCoupling", Core::Conditions::BioGrCoupling, true,
      Core::Conditions::geometry_type_surface);

  surfbiogr.add_component(entry<int>("coupling_id", {.description = "coupling_id"}));
  condlist.push_back(surfbiogr);
}


void Inpar::ReducedLung::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& redawdyn = list.sublist("REDUCED DIMENSIONAL AIRWAYS DYNAMIC", false, "");

  setStringToIntegralParameter<RedAirwaysDyntype>("DYNAMICTYPE", "OneStepTheta",
      "OneStepTheta Scheme", tuple<std::string>("OneStepTheta"),
      tuple<RedAirwaysDyntype>(one_step_theta), &redawdyn);

  setStringToIntegralParameter<RedAirwaysDyntype>("SOLVERTYPE", "Linear", "Solver type",
      tuple<std::string>("Linear", "Nonlinear"), tuple<RedAirwaysDyntype>(linear, nonlinear),
      &redawdyn);

  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", &redawdyn);
  Core::Utils::int_parameter("NUMSTEP", 0, "Number of Time Steps", &redawdyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", &redawdyn);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", &redawdyn);
  Core::Utils::double_parameter("THETA", 1.0, "One-step-theta time integration factor", &redawdyn);

  Core::Utils::int_parameter("MAXITERATIONS", 1, "maximum iteration steps", &redawdyn);
  Core::Utils::double_parameter("TOLERANCE", 1.0E-6, "tolerance", &redawdyn);

  // number of linear solver used for reduced dimensional airways dynamic
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "number of linear solver used for reduced dim arterial dynamics", &redawdyn);

  Core::Utils::bool_parameter(
      "SOLVESCATRA", "no", "Flag to (de)activate solving scalar transport in blood", &redawdyn);

  Core::Utils::bool_parameter("COMPAWACINTER", "no",
      "Flag to (de)activate computation of airway-acinus interdependency", &redawdyn);

  Core::Utils::bool_parameter("CALCV0PRESTRESS", "no",
      "Flag to (de)activate initial acini volume adjustment with pre-stress condition", &redawdyn);

  Core::Utils::double_parameter("TRANSPULMPRESS", 800.0,
      "Transpulmonary pressure needed for recalculation of acini volumes", &redawdyn);
}



void Inpar::ReducedLung::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_red_to_3d_bc(
      "DESIGN NODE REDUCED D To 3D FLOW COUPLING CONDITIONS", "Art_redD_3D_CouplingCond",
      "Artery reduced D 3D coupling condition", Core::Conditions::ArtRedTo3DCouplingCond, true,
      Core::Conditions::geometry_type_point);

  art_red_to_3d_bc.add_component(entry<int>("ConditionID"));
  art_red_to_3d_bc.add_component(selection<std::string>("CouplingType", {"forced", "absorbing"},
      {.description = "coupling type", .default_value = "forced"}));
  art_red_to_3d_bc.add_component(selection<std::string>("ReturnedVariable", {"pressure", "flow"},
      {.description = "returned variable", .default_value = "pressure"}));
  art_red_to_3d_bc.add_component(entry<double>("Tolerance"));
  art_red_to_3d_bc.add_component(entry<int>("MaximumIterations"));

  condlist.push_back(art_red_to_3d_bc);

  /*--------------------------------------------------------------------*/
  // 3-D/reduced-D coupling boundary condition
  Core::Conditions::ConditionDefinition art_3d_to_red_bc(
      "DESIGN SURF 3D To REDUCED D FLOW COUPLING CONDITIONS", "Art_3D_redD_CouplingCond",
      "Artery 3D reduced D coupling condition", Core::Conditions::Art3DToRedCouplingCond, true,
      Core::Conditions::geometry_type_surface);

  art_3d_to_red_bc.add_component(entry<int>("ConditionID"));
  art_3d_to_red_bc.add_component(selection<std::string>("ReturnedVariable", {"pressure", "flow"},
      {.description = "returned variable", .default_value = "flow"}));
  art_3d_to_red_bc.add_component(entry<double>("Tolerance"));
  art_3d_to_red_bc.add_component(entry<int>("MaximumIterations"));

  condlist.push_back(art_3d_to_red_bc);

  /*--------------------------------------------------------------------*/
  // Coupling of 3D tissue models and reduced-D airway tree

  Core::Conditions::ConditionDefinition surfredairtis("DESIGN SURF TISSUE REDAIRWAY CONDITIONS",
      "SurfaceNeumann", "tissue RedAirway coupling surface condition",
      Core::Conditions::RedAirwayTissue, true, Core::Conditions::geometry_type_surface);

  surfredairtis.add_component(entry<int>("coupling_id"));

  condlist.push_back(surfredairtis);


  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways

  Core::Conditions::ConditionDefinition noderedairtis("DESIGN NODE TISSUE REDAIRWAY CONDITIONS",
      "RedAirwayPrescribedCond", "tissue RedAirway coupling node condition",
      Core::Conditions::RedAirwayNodeTissue, true, Core::Conditions::geometry_type_point);

  noderedairtis.add_component(entry<int>("coupling_id"));

  condlist.push_back(noderedairtis);



  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_in_bc(
      "DESIGN NODE Reduced D AIRWAYS PRESCRIBED CONDITIONS", "RedAirwayPrescribedCond",
      "Reduced d airway prescribed boundary condition", Core::Conditions::RedAirwayPrescribedCond,
      true, Core::Conditions::geometry_type_point);

  raw_in_bc.add_component(selection<std::string>("boundarycond",
      {"flow", "pressure", "switchFlowPressure", "VolumeDependentPleuralPressure"},
      {.description = "boundary condition type", .default_value = "flow"}));

  // reduced airway inlet components
  raw_in_bc.add_component(entry<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  raw_in_bc.add_component(
      entry<std::vector<Noneable<int>>>("curve", {.description = "curve", .size = 2}));
  raw_in_bc.add_component(entry<std::vector<Noneable<int>>>(
      "funct", {.description = "function id", .default_value = std::vector{none<int>}, .size = 1}));

  condlist.push_back(raw_in_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed BC for reduced dimensional airways switching between different types of boundary
  // conditions

  Core::Conditions::ConditionDefinition raw_in_switch_bc(
      "DESIGN NODE Reduced D AIRWAYS SWITCH FLOW PRESSURE CONDITIONS",
      "RedAirwaySwitchFlowPressureCond", "Reduced d airway switch flow pressure boundary condition",
      Core::Conditions::RedAirwayPrescribedSwitchCond, true, Core::Conditions::geometry_type_point);

  raw_in_switch_bc.add_component(entry<int>("FUNCT_ID_FLOW"));
  raw_in_switch_bc.add_component(entry<int>("FUNCT_ID_PRESSURE"));
  raw_in_switch_bc.add_component(entry<int>("FUNCT_ID_PRESSURE_ACTIVE"));

  condlist.push_back(raw_in_switch_bc);

  /*--------------------------------------------------------------------*/
  // Prescribed volume dependent pleural pressure for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_volPpl_bc(
      "DESIGN LINE REDUCED D AIRWAYS VOL DEPENDENT PLEURAL PRESSURE CONDITIONS",
      "RedAirwayVolDependentPleuralPressureCond",
      "Reduced D airways volume-dependent peural pressure condition",
      Core::Conditions::RedAirwayVolDependentPleuralPressureCond, true,
      Core::Conditions::geometry_type_line);

  raw_volPpl_bc.add_component(selection<std::string>("TYPE",
      {"Linear_Polynomial", "Linear_Exponential", "Linear_Ogden", "Nonlinear_Polynomial",
          "Nonlinear_Exponential", "Nonlinear_Ogden"},
      {.description = "type", .default_value = "Linear_Exponential"}));

  raw_volPpl_bc.add_component(entry<double>("TLC"));
  raw_volPpl_bc.add_component(entry<double>("RV"));

  raw_volPpl_bc.add_component(entry<double>("P_PLEURAL_0"));
  raw_volPpl_bc.add_component(entry<double>("P_PLEURAL_LIN"));
  raw_volPpl_bc.add_component(entry<double>("P_PLEURAL_NONLIN"));
  raw_volPpl_bc.add_component(entry<double>("TAU"));

  // raw_volPpl_bc_components
  raw_volPpl_bc.add_component(
      entry<std::vector<double>>("VAL", {.description = "value", .size = 1}));
  raw_volPpl_bc.add_component(
      entry<std::vector<Noneable<int>>>("curve", {.description = "curve", .size = 1}));

  condlist.push_back(raw_volPpl_bc);

  /*--------------------------------------------------------------------*/
  // Evaluate lung volume condition for reduced dimensional airways

  Core::Conditions::ConditionDefinition raw_eval_lungV_bc(
      "DESIGN LINE REDUCED D AIRWAYS EVALUATE LUNG VOLUME CONDITIONS", "RedAirwayEvalLungVolCond",
      "Reduced D airways evaluate lung volume condition",
      Core::Conditions::RedAirwayEvalLungVolCond, true, Core::Conditions::geometry_type_line);


  condlist.push_back(raw_eval_lungV_bc);


  /*--------------------------------------------------------------------*/
  // Impedance condition

  Core::Conditions::ConditionDefinition impedancebc("DESIGN SURF IMPEDANCE CONDITIONS",
      "ImpedanceCond", "Impedance boundary condition", Core::Conditions::ImpedanceCond, true,
      Core::Conditions::geometry_type_surface);

  impedancebc.add_component(entry<int>("ConditionID"));
  impedancebc.add_component(
      selection<std::string>("TYPE", {"windkessel", "resistive", "pressure_by_funct"},
          {.description = "type", .default_value = "windkessel"}));
  impedancebc.add_component(entry<double>("R1"));
  impedancebc.add_component(entry<double>("R2"));
  impedancebc.add_component(entry<double>("C"));
  impedancebc.add_component(entry<double>("TIMEPERIOD"));
  impedancebc.add_component(entry<int>("FUNCT"));

  condlist.push_back(impedancebc);
}

FOUR_C_NAMESPACE_CLOSE

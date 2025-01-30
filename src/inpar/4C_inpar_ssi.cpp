// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_ssi.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::SSI::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& ssidyn =
      list.sublist("SSI CONTROL", false, "Control parameters for scatra structure interaction");

  // Output type
  Core::Utils::double_parameter(
      "RESTARTEVERYTIME", 0, "write restart possibility every RESTARTEVERY steps", &ssidyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", &ssidyn);
  // Time loop control
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", &ssidyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "total simulation time", &ssidyn);
  Core::Utils::double_parameter("TIMESTEP", -1, "time step size dt", &ssidyn);
  Core::Utils::bool_parameter(
      "DIFFTIMESTEPSIZE", "No", "use different step size for scatra and solid", &ssidyn);
  Core::Utils::double_parameter("RESULTSEVERYTIME", 0, "increment for writing solution", &ssidyn);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "increment for writing solution", &ssidyn);
  Core::Utils::int_parameter("ITEMAX", 10, "maximum number of iterations over fields", &ssidyn);
  Core::Utils::bool_parameter("SCATRA_FROM_RESTART_FILE", "No",
      "read scatra result from restart files (use option 'restartfromfile' during execution of "
      "4C)",
      &ssidyn);
  Core::Utils::string_parameter(
      "SCATRA_FILENAME", "nil", "Control-file name for reading scatra results in SSI", &ssidyn);

  // Type of coupling strategy between the two fields
  setStringToIntegralParameter<FieldCoupling>("FIELDCOUPLING", "volume_matching",
      "Type of coupling strategy between fields",
      tuple<std::string>("volume_matching", "volume_nonmatching", "boundary_nonmatching",
          "volumeboundary_matching"),
      tuple<FieldCoupling>(FieldCoupling::volume_match, FieldCoupling::volume_nonmatch,
          FieldCoupling::boundary_nonmatch, FieldCoupling::volumeboundary_match),
      &ssidyn);

  // Coupling strategy for SSI solvers
  setStringToIntegralParameter<SolutionSchemeOverFields>("COUPALGO", "ssi_IterStagg",
      "Coupling strategies for SSI solvers",
      tuple<std::string>("ssi_OneWay_ScatraToSolid", "ssi_OneWay_SolidToScatra",
          //                                "ssi_SequStagg_ScatraToSolid",
          //                                "ssi_SequStagg_SolidToScatra",
          "ssi_IterStagg", "ssi_IterStaggFixedRel_ScatraToSolid",
          "ssi_IterStaggFixedRel_SolidToScatra", "ssi_IterStaggAitken_ScatraToSolid",
          "ssi_IterStaggAitken_SolidToScatra", "ssi_Monolithic"),
      tuple<SolutionSchemeOverFields>(SolutionSchemeOverFields::ssi_OneWay_ScatraToSolid,
          SolutionSchemeOverFields::ssi_OneWay_SolidToScatra,
          //                                ssi_SequStagg_ScatraToSolid,
          //                                ssi_SequStagg_SolidToScatra,
          SolutionSchemeOverFields::ssi_IterStagg,
          SolutionSchemeOverFields::ssi_IterStaggFixedRel_ScatraToSolid,
          SolutionSchemeOverFields::ssi_IterStaggFixedRel_SolidToScatra,
          SolutionSchemeOverFields::ssi_IterStaggAitken_ScatraToSolid,
          SolutionSchemeOverFields::ssi_IterStaggAitken_SolidToScatra,
          SolutionSchemeOverFields::ssi_Monolithic),
      &ssidyn);

  // type of scalar transport time integration
  setStringToIntegralParameter<ScaTraTimIntType>("SCATRATIMINTTYPE", "Standard",
      "scalar transport time integration type is needed to instantiate correct scalar transport "
      "time integration scheme for ssi problems",
      tuple<std::string>("Standard", "Cardiac_Monodomain", "Elch"),
      tuple<ScaTraTimIntType>(
          ScaTraTimIntType::standard, ScaTraTimIntType::cardiac_monodomain, ScaTraTimIntType::elch),
      &ssidyn);

  // Restart from Structure problem instead of SSI
  Core::Utils::bool_parameter("RESTART_FROM_STRUCTURE", "no",
      "restart from structure problem (e.g. from prestress calculations) instead of ssi", &ssidyn);

  // Adaptive time stepping
  Core::Utils::bool_parameter(
      "ADAPTIVE_TIMESTEPPING", "no", "flag for adaptive time stepping", &ssidyn);

  // do redistribution by binning of solid mechanics discretization (scatra dis is cloned from solid
  // dis for volume_matching and volumeboundary_matching)
  Core::Utils::bool_parameter("REDISTRIBUTE_SOLID", "No",
      "redistribution by binning of solid mechanics discretization", &ssidyn);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned SSI */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynpart = ssidyn.sublist("PARTITIONED", false,
      "Partitioned Structure Scalar Interaction\n"
      "Control section for partitioned SSI");

  // Solver parameter for relaxation of iterative staggered partitioned SSI
  Core::Utils::double_parameter(
      "MAXOMEGA", 10.0, "largest omega allowed for Aitken relaxation", &ssidynpart);
  Core::Utils::double_parameter(
      "MINOMEGA", 0.1, "smallest omega allowed for Aitken relaxation", &ssidynpart);
  Core::Utils::double_parameter("STARTOMEGA", 1.0, "fixed relaxation parameter", &ssidynpart);

  // convergence tolerance of outer iteration loop
  Core::Utils::double_parameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration within partitioned SSI", &ssidynpart);

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSI */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynmono = ssidyn.sublist("MONOLITHIC", false,
      "Monolithic Structure Scalar Interaction\n"
      "Control section for monolithic SSI");

  // convergence tolerances of Newton-Raphson iteration loop
  Core::Utils::double_parameter("ABSTOLRES", 1.e-14,
      "absolute tolerance for deciding if global residual of nonlinear problem is already zero",
      &ssidynmono);
  Core::Utils::double_parameter("CONVTOL", 1.e-6,
      "tolerance for convergence check of Newton-Raphson iteration within monolithic SSI",
      &ssidynmono);

  // ID of linear solver for global system of equations
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "ID of linear solver for global system of equations", &ssidynmono);

  // type of global system matrix in global system of equations
  setStringToIntegralParameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag", "local"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::local),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_STRUCTURE", "none",
      "flag for equilibration of structural equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_SCATRA", "none",
      "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &ssidynmono);

  Core::Utils::bool_parameter("PRINT_MAT_RHS_MAP_MATLAB", "no",
      "print system matrix, rhs vector, and full map to matlab readable file after solution of "
      "time step",
      &ssidynmono);

  Core::Utils::double_parameter("RELAX_LIN_SOLVER_TOLERANCE", 1.0,
      "relax the tolerance of the linear solver in case it is an iterative solver by scaling the "
      "convergence tolerance with factor RELAX_LIN_SOLVER_TOLERANCE",
      &ssidynmono);

  Core::Utils::int_parameter("RELAX_LIN_SOLVER_STEP", -1,
      "relax the tolerance of the linear solver within the first RELAX_LIN_SOLVER_STEP steps",
      &ssidynmono);

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with manifold */
  /*----------------------------------------------------------------------*/

  Teuchos::ParameterList& ssidynmanifold = ssidyn.sublist("MANIFOLD", false,
      "Monolithic Structure Scalar Interaction with additional scalar transport on manifold");

  Core::Utils::bool_parameter(
      "ADD_MANIFOLD", "no", "activate additional manifold?", &ssidynmanifold);

  Core::Utils::bool_parameter("MESHTYING_MANIFOLD", "no",
      "activate meshtying between all manifold fields in case they intersect?", &ssidynmanifold);

  setStringToIntegralParameter<Inpar::ScaTra::InitialField>("INITIALFIELD", "zero_field",
      "Initial field for scalar transport on manifold",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<Inpar::ScaTra::InitialField>(Inpar::ScaTra::initfield_zero_field,
          Inpar::ScaTra::initfield_field_by_function, Inpar::ScaTra::initfield_field_by_condition),
      &ssidynmanifold);

  Core::Utils::int_parameter("INITFUNCNO", -1,
      "function number for scalar transport on manifold initial field", &ssidynmanifold);

  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "linear solver for scalar transport on manifold", &ssidynmanifold);

  Core::Utils::bool_parameter("OUTPUT_INFLOW", "no",
      "write output of inflow of scatra manifold - scatra coupling into scatra manifold to csv "
      "file",
      &ssidynmanifold);

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with elch */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynelch = ssidyn.sublist(
      "ELCH", false, "Monolithic Structure Scalar Interaction with Elch as SCATRATIMINTTYPE");
  Core::Utils::bool_parameter("INITPOTCALC", "No",
      "Automatically calculate initial field for electric potential", &ssidynelch);
}

/*--------------------------------------------------------------------
--------------------------------------------------------------------*/
void Inpar::SSI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;


  /*--------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition linessiplain("DESIGN SSI COUPLING LINE CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiplain("DESIGN SSI COUPLING SURF CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssiplain("DESIGN SSI COUPLING VOL CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssiplain = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(entry<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssiplain(linessiplain);
  make_ssiplain(surfssiplain);

  /*--------------------------------------------------------------------*/
  //! set solid dofset on scatra discretization
  Core::Conditions::ConditionDefinition linessi("DESIGN SSI COUPLING SOLIDTOSCATRA LINE CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssi("DESIGN SSI COUPLING SOLIDTOSCATRA SURF CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssi("DESIGN SSI COUPLING SOLIDTOSCATRA VOL CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssi = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(entry<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssi(linessi);
  make_ssi(surfssi);
  make_ssi(volssi);

  /*--------------------------------------------------------------------*/
  //! set scatra dofset on solid discretization
  Core::Conditions::ConditionDefinition linessi2(
      "DESIGN SSI COUPLING SCATRATOSOLID LINE CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssi2(
      "DESIGN SSI COUPLING SCATRATOSOLID SURF CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssi2("DESIGN SSI COUPLING SCATRATOSOLID VOL CONDITIONS",
      "SSICouplingScatraToSolid", "SSI Coupling ScatraToSolid",
      Core::Conditions::SSICouplingScatraToSolid, true, Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssi2 = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(entry<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssi2(linessi2);
  make_ssi2(surfssi2);
  make_ssi2(volssi2);


  /*--------------------------------------------------------------------*/
  // set ScaTra-Structure interaction interface meshtying condition
  Core::Conditions::ConditionDefinition pointssiinterfacemeshtying(
      "DESIGN SSI INTERFACE MESHTYING POINT CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linessiinterfacemeshtying(
      "DESIGN SSI INTERFACE MESHTYING LINE CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiinterfacemeshtying(
      "DESIGN SSI INTERFACE MESHTYING SURF CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_surface);

  // equip condition definitions with input file line components
  //
  // REMARK: it would be cleaner to also set a reference to the structural meshtying condition here
  // and not only to the S2ICoupling condition. Of course, then also the structural meshtying should
  // be used which could/should be the long-term goal. However, to date, a simple structural
  // meshtying version for matching node is implemented within the SSI framework and therefore no
  // reference is necessary.

  // insert input file line components into condition definitions
  const auto make_ssiinterfacemeshtying = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(entry<int>("ConditionID"));
    cond.add_component(selection<int>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface_side"}));
    cond.add_component(entry<int>("S2I_KINETICS_ID"));

    condlist.push_back(cond);
  };

  make_ssiinterfacemeshtying(pointssiinterfacemeshtying);
  make_ssiinterfacemeshtying(linessiinterfacemeshtying);
  make_ssiinterfacemeshtying(surfssiinterfacemeshtying);

  /*--------------------------------------------------------------------*/
  // condition, where additional scatra field on manifold is created
  Core::Conditions::ConditionDefinition ssisurfacemanifold("DESIGN SSI MANIFOLD SURF CONDITIONS",
      "SSISurfaceManifold", "scalar transport on manifold", Core::Conditions::SSISurfaceManifold,
      true, Core::Conditions::geometry_type_surface);

  ssisurfacemanifold.add_component(entry<int>("ConditionID"));
  ssisurfacemanifold.add_component(selection<int>("ImplType",
      {{"Undefined", Inpar::ScaTra::impltype_undefined}, {"Standard", Inpar::ScaTra::impltype_std},
          {"ElchElectrode", Inpar::ScaTra::impltype_elch_electrode},
          {"ElchDiffCond", Inpar::ScaTra::impltype_elch_diffcond}},
      {.description = "implementation type"}));
  ssisurfacemanifold.add_component(entry<double>("thickness"));

  condlist.emplace_back(ssisurfacemanifold);

  /*--------------------------------------------------------------------*/
  // initial field by condition for scatra on manifold
  Core::Conditions::ConditionDefinition surfmanifoldinitfields(
      "DESIGN SURF SCATRA MANIFOLD INITIAL FIELD CONDITIONS", "ScaTraManifoldInitfield",
      "Surface ScaTra Manifold Initfield", Core::Conditions::SurfaceInitfield, false,
      Core::Conditions::geometry_type_surface);

  surfmanifoldinitfields.add_component(
      selection<std::string>("FIELD", {"ScaTra"}, {.description = "init field"}));
  surfmanifoldinitfields.add_component(entry<int>("FUNCT"));

  condlist.emplace_back(surfmanifoldinitfields);

  /*--------------------------------------------------------------------*/
  // kinetics condition for flux scatra <-> scatra on manifold
  Core::Conditions::ConditionDefinition surfmanifoldkinetics(
      "DESIGN SSI MANIFOLD KINETICS SURF CONDITIONS", "SSISurfaceManifoldKinetics",
      "kinetics model for coupling scatra <-> scatra on manifold",
      Core::Conditions::SSISurfaceManifoldKinetics, true, Core::Conditions::geometry_type_surface);

  {
    surfmanifoldkinetics.add_component(entry<int>("ConditionID"));
    surfmanifoldkinetics.add_component(entry<int>("ManifoldConditionID"));

    using namespace Core::IO::InputSpecBuilders;

    surfmanifoldkinetics.add_component(one_of({
        all_of({
            selection<int>("KINETIC_MODEL", {{"ConstantInterfaceResistance",
                                                Inpar::S2I::kinetics_constantinterfaceresistance}}),
            entry<std::vector<int>>("ONOFF", {.size = 2}),
            entry<double>("RESISTANCE"),
            entry<int>("E-"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL",
                {{"Butler-VolmerReduced", Inpar::S2I::kinetics_butlervolmerreduced}}),
            entry<int>("NUMSCAL"),
            entry<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            entry<int>("E-"),
            entry<double>("K_R"),
            entry<double>("ALPHA_A"),
            entry<double>("ALPHA_C"),
        }),
        selection<int>(
            "KINETIC_MODEL", {{"NoInterfaceFlux", Inpar::S2I::kinetics_nointerfaceflux}}),
    }));
  }

  condlist.emplace_back(surfmanifoldkinetics);

  /*--------------------------------------------------------------------*/
  // Dirichlet conditions for scatra on manifold
  Core::Conditions::ConditionDefinition pointmanifolddirichlet(
      "DESIGN POINT MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linemanifolddirichlet(
      "DESIGN LINE MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Line Dirichlet",
      Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfmanifolddirichlet(
      "DESIGN SURF MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Surface Dirichlet",
      Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);

  const auto add_dirichlet_manifold_components =
      [](Core::Conditions::ConditionDefinition& definition)
  {
    definition.add_component(entry<int>("NUMDOF"));
    definition.add_component(entry<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    definition.add_component(entry<std::vector<double>>(
        "VAL", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    definition.add_component(entry<std::vector<Noneable<int>>>(
        "FUNCT", {.description = "", .size = from_parameter<int>("NUMDOF")}));
  };

  {
    add_dirichlet_manifold_components(pointmanifolddirichlet);
    add_dirichlet_manifold_components(linemanifolddirichlet);
    add_dirichlet_manifold_components(surfmanifolddirichlet);
  }

  condlist.push_back(pointmanifolddirichlet);
  condlist.push_back(linemanifolddirichlet);
  condlist.push_back(surfmanifolddirichlet);

  /*--------------------------------------------------------------------*/
  // set ScaTra-Structure Interaction interface contact condition
  Core::Conditions::ConditionDefinition linessiinterfacecontact(
      "DESIGN SSI INTERFACE CONTACT LINE CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiinterfacecontact(
      "DESIGN SSI INTERFACE CONTACT SURF CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_surface);

  // insert input file line components into condition definitions
  const auto make_ssiinterfacecontact = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(entry<int>("ConditionID"));
    cond.add_component(selection<int>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface_side"}));
    cond.add_component(entry<int>("S2I_KINETICS_ID"));
    cond.add_component(entry<int>("CONTACT_CONDITION_ID"));

    condlist.push_back(cond);
  };

  make_ssiinterfacecontact(linessiinterfacecontact);
  make_ssiinterfacecontact(surfssiinterfacecontact);
}

FOUR_C_NAMESPACE_CLOSE

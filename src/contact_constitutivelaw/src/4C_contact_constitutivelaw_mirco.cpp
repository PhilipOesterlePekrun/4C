// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_mirco.hpp"

#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_par_bundle.hpp"

#ifdef FOUR_C_WITH_MIRCO

#include <mirco_evaluate.h>
#include <mirco_kokkostypes.h>
#include <mirco_topology.h>
#include <mirco_topologyutilities.h>

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams::MircoConstitutiveLawParams(
    const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container),
      finite_difference_fraction_(container.get<double>("FiniteDifferenceFraction")),
      active_gap_tolerance_(container.get<double>("ActiveGapTolerance"))
{
  const auto& parameters = container.group("parameters");
  const auto& geometricalParameters = parameters.group("geometrical_parameters");
  const auto& materialParameters = parameters.group("material_parameters");

  first_mat_id = geometricalParameters.get<int>("FirstMatID");
  second_mat_id = materialParameters.get<int>("SecondMatID");

  // retrieve material parameters
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW(
        "An attempt to access the list of materials in the instance of the global problem returned "
        "a null pointer.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");
  // retrieve validated input line of material ID in question
  const auto& firstmat = Global::Problem::instance(probinst)
                             ->materials()
                             ->parameter_by_id(first_mat_id)
                             ->raw_parameters();
  const auto& secondmat = Global::Problem::instance(probinst)
                              ->materials()
                              ->parameter_by_id(second_mat_id)
                              ->raw_parameters();

  auto exportVisualization = container.get_or<bool>("ExportVisualization", std::nullopt)
                                 std::optional<std::string>
                                     exportVisualizationPath;
  if (exportVisualization && exportVisualization.value())
    exportVisualizationPath = container.get<std::string>(root, "ExportVisualizationPath");
  else
    exportVisualizationPath = std::nullopt;

  if (rget<bool>(root, "RandomTopologyFlag"))
  {
    *this = InputParameters(firstmat.get<double>("YOUNG"), secondmat.get<double>("YOUNG"),
        firstmat.get<double>("NUE"), secondmat.get<double>("NUE"),
        geometricalParameters.get<double>("Tolerance"), geometricalParameters.get<double>("Delta"),
        geometricalParameters.get<double>("LateralLength"),
        rget<int>(geometricalParameters, "Resolution"),
        geometricalParameters.get<double>("InitialTopologyStdDeviation"),
        geometricalParameters.get<double>("HurstExponent"), container.get<int>("MaxIteration"),
        container.get<bool>("WarmStartingFlag"), container.get<bool>("PressureGreenFunFlag"),
        container.get<bool>("RandomSeedFlag"),
        container.get_or<int>("RandomGeneratorSeed", std::nullopt), exportVisualizationPath);
  }
  else
  {
    std::string topology_file_path = rget<std::string>(root, "TopologyFilePath");
    // If the path is relative, it is relative to the input (.yaml) file
    std::filesystem::path new_path = topology_file_path;
    if (new_path.is_relative())
      new_path = std::filesystem::path(inputFileName).parent_path() / new_path;
    topology_file_path = new_path.string();

    *this = InputParameters(rget<double>(firstmat.get<double>("YOUNG"), secondmat.get<double>("YOUNG"),
        firstmat.get<double>("NUE"), secondmat.get<double>("NUE"),
        geometricalParameters.get<double>("Tolerance"), geometricalParameters.get<double>("Delta"),
        geometricalParameters.get<double>("LateralLength"), topology_file_path,
        container.get<int>("MaxIteration"), container.get<bool>("WarmStartingFlag"),
        container.get_or<int>("RandomGeneratorSeed", std::nullopt), exportVisualizationPath);
  }

  Kokkos::View<int*> x("x", 1);

  Kokkos::parallel_for("test", 1, KOKKOS_LAMBDA(const int i) { x(i) = 42; });

  Kokkos::fence();

  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  std::cout << x_h(0) << '\n';
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::MircoConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params)
    : params_(std::move(params))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW(
        "The Evaluate function can only operate on active nodes. With a current gap = {} and an "
        "initial offset = {}, this node is inactive though.",
        gap, params_.get_offset());
  }

  if (-(gap + params_.get_offset()) < params_.active_gap_tolerance)
  {
    return 0.0;
  }

  const RoughNode* roughNode = dynamic_cast<const RoughNode*>(cnode);
  auto topology = *roughNode->get_topology();

  MIRCO::ViewVector_d meshgrid = MIRCO::CreateMeshgrid(params_.N, params_.grid_size);
  const double topologyMax = MIRCO::GetMax(params_.topology);

  double pressure, contact_area_fraction;
  MIRCO::Evaluate(pressure, contact_area_fraction, -(gap + params_.get_offset()),
      params_);  //, topologyMax, meshgrid);

  return (-1 * pressure);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW(
        "The Evaluate function can only operate on active nodes. With a current gap = {} and an "
        "initial offset = {}, this node is inactive though.",
        gap, params_.get_offset());
  }

  if (-(gap + params_.get_offset()) < params_.active_gap_tolerance)
  {
    return 0.0;
  }

  const RoughNode* roughNode = dynamic_cast<const RoughNode*>(cnode);
  auto topology = *roughNode->get_topology();

  double pressure1 = 0.0;
  double pressure2 = 0.0;
  double contact_area_fraction = 0.0;
  // using backward difference approach
  MIRCO::Evaluate(pressure1, contact_area_fraction, -1.0 * (gap + params_.get_offset()),
      params_.lateral_length, params_.grid_size, params_.tolerance, params_.max_iteration,
      params_.composite_youngs, params_.warm_starting_flag, params_.elastic_compliance_correction,
      topology, roughNode->get_max_topology_height(), *params_.mesh_grid,
      params_.pressure_green_fun_flag);
  MIRCO::Evaluate(pressure2, contact_area_fraction,
      -(1 - params_.finite_difference_fraction) * (gap + params_.get_offset()),
      params_.lateral_length, params_.grid_size, params_.tolerance, params_.max_iteration,
      params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.elastic_compliance_correction, topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());
  return ((pressure1 - pressure2) /
          (-(params_.get_finite_difference_fraction()) * (gap + params_.get_offset())));
}

FOUR_C_NAMESPACE_CLOSE

#endif

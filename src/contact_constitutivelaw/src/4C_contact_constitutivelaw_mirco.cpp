// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_mirco.hpp"

#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_par_bundle.hpp"

#ifdef FOUR_C_WITH_MIRCO

#include <mirco_evaluate.h>
#include <mirco_kokkostypes.h>
#include <mirco_shapefactors.h>
#include <mirco_topology.h>
#include <mirco_topologyutilities.h>

#include <filesystem>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::string resolve_mirco_topology_file_path(const std::string& topology_file_path)
  {
    if (topology_file_path.empty()) return {};

    std::filesystem::path path(topology_file_path);
    if (path.is_relative())
    {
      const auto output_control = Global::Problem::instance()->output_control_file();
      if (output_control == nullptr)
        FOUR_C_THROW(
            "Cannot resolve relative MIRCO topology file path '{}' because no input file is "
            "registered in the output control.",
            topology_file_path);

      path = std::filesystem::path(output_control->input_file_name()).parent_path() / path;
    }

    return path.lexically_normal().string();
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams::MircoConstitutiveLawParams(
    const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container),
      firstmatid_(container.get<int>("FirstMatID")),
      secondmatid_(container.get<int>("SecondMatID")),
      lateral_length_(container.get<double>("LateralLength")),
      resolution_(container.get<int>("Resolution")),
      pressure_green_fun_flag_(container.get<bool>("PressureGreenFunFlag")),
      random_topology_flag_(container.get<bool>("RandomTopologyFlag")),
      random_seed_flag_(container.get<bool>("RandomSeedFlag")),
      random_generator_seed_(container.get<int>("RandomGeneratorSeed")),
      tolerance_(container.get<double>("Tolerance")),
      max_iteration_(container.get<int>("MaxIteration")),
      warm_starting_flag_(container.get<bool>("WarmStartingFlag")),
      finite_difference_fraction_(container.get<double>("FiniteDifferenceFraction")),
      active_gap_tolerance_(container.get<double>("ActiveGapTolerance")),
      topology_file_path_(
          resolve_mirco_topology_file_path(container.get<std::string>("TopologyFilePath")))
{
  this->set_parameters();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw::MircoConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params)
    : params_(std::move(params))
{
}

void CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams::set_parameters()
{
  // retrieve problem instance to read from
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
                             ->parameter_by_id(get_first_mat_id())
                             ->raw_parameters();
  const auto& secondmat = Global::Problem::instance(probinst)
                              ->materials()
                              ->parameter_by_id(get_second_mat_id())
                              ->raw_parameters();

  const double E1 = firstmat.get<double>("YOUNG");
  const double E2 = secondmat.get<double>("YOUNG");
  const double nu1 = firstmat.get<double>("NUE");
  const double nu2 = secondmat.get<double>("NUE");

  // Composite Young's modulus
  composite_youngs_ = pow(((1 - pow(nu1, 2)) / E1 + (1 - pow(nu2, 2)) / E2), -1);

  int ngrid = 0;
  if (!topology_file_path_.empty())
  {
    if (!std::filesystem::is_regular_file(topology_file_path_))
      FOUR_C_THROW(
          "MIRCO topology file '{}' does not exist or is not a regular file.", topology_file_path_);

    const auto topology = MIRCO::CreateSurfaceFromFile(topology_file_path_);
    if (topology.extent(0) == 0 || topology.extent(0) != topology.extent(1))
      FOUR_C_THROW("MIRCO topology must be a non-empty square matrix, but '{}' produced {} x {}.",
          topology_file_path_, topology.extent(0), topology.extent(1));
    ngrid = static_cast<int>(topology.extent(0));
  }
  else
  {
    if (resolution_ < 1 || resolution_ > 8)
      FOUR_C_THROW("MIRCO Resolution must be between 1 and 8 when no TopologyFilePath is given.");
    ngrid = (1 << resolution_) + 1;
  }

  grid_size_ = lateral_length_ / ngrid;

  const double ShapeFactor = MIRCO::getShapeFactor(ngrid, pressure_green_fun_flag_);

  elastic_compliance_correction_ = lateral_length_ * composite_youngs_ / ShapeFactor;

  meshgrid_ = MIRCO::CreateMeshgrid(ngrid, grid_size_);
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

  if (-(gap + params_.get_offset()) < params_.get_active_gap_tolerance())
  {
    return 0.0;
  }

  const RoughNode* roughNode = dynamic_cast<const RoughNode*>(cnode);
  auto topology = *roughNode->get_topology();

  double pressure = 0.0;
  double contact_area_fraction = 0.0;
  MIRCO::Evaluate(pressure, contact_area_fraction, -(gap + params_.get_offset()),
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());

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

  if (-(gap + params_.get_offset()) < params_.get_active_gap_tolerance())
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
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());
  MIRCO::Evaluate(pressure2, contact_area_fraction,
      -(1 - params_.get_finite_difference_fraction()) * (gap + params_.get_offset()),
      params_.get_lateral_length(), params_.get_grid_size(), params_.get_tolerance(),
      params_.get_max_iteration(), params_.get_composite_youngs(), params_.get_warm_starting_flag(),
      params_.get_compliance_correction(), topology, roughNode->get_max_topology_height(),
      *params_.get_mesh_grid(), params_.get_pressure_green_fun_flag());
  return ((pressure1 - pressure2) /
          (-(params_.get_finite_difference_fraction()) * (gap + params_.get_offset())));
}

FOUR_C_NAMESPACE_CLOSE

#endif

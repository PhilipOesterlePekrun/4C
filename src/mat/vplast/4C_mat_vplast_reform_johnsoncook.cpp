// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_vplast_reform_johnsoncook.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::PAR::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      strain_rate_prefac_(matdata.parameters.get<double>("STRAIN_RATE_PREFAC")),
      strain_rate_exp_fac_(matdata.parameters.get<double>("STRAIN_RATE_EXP_FAC")),
      init_yield_strength_(matdata.parameters.get<double>("INIT_YIELD_STRENGTH")),
      isotrop_harden_prefac_(matdata.parameters.get<double>("ISOTROP_HARDEN_PREFAC")),
      isotrop_harden_exp_(matdata.parameters.get<double>("ISOTROP_HARDEN_EXP"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    Core::Mat::PAR::Parameter* params)
    : Mat::Viscoplastic::Law(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_stress_ratio(
    const double equiv_stress, const double equiv_plastic_strain)
{
  // extract yield strength from the plastic strain and the material parameters
  const double yield_strength =
      (parameter()->init_yield_strength() +
          parameter()->isotrop_harden_prefac() *
              std::pow(equiv_plastic_strain, parameter()->isotrop_harden_exp()));

  return equiv_stress / yield_strength;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const bool update_hist_var)
{
  // unwrap variables
  const double p = parameter()->strain_rate_pre_fac();        // prefactor of plastic strain rate
  const double e = 1.0 / parameter()->strain_rate_exp_fac();  // exponent of plastic strain rate

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    throw std::runtime_error(
        "ERROR 1: Negative Plastic Strain " + std::to_string(equiv_plastic_strain));
  }

  // compute the viscoplastic strain rate; first we set it to 0
  double equiv_plastic_strain_rate = 0.0;

  // stress ratio
  double stress_ratio = evaluate_stress_ratio(equiv_stress, equiv_plastic_strain);

  // then we check the yield condition
  if (stress_ratio >= 1.0)
  {
    // check if characteristic term too large, throw error overflow error if so
    if (std::log(dt) + std::log(p) + e * (stress_ratio - 1.0) > std::log(10.0 + p * dt))
    {
      throw std::overflow_error(
          "ERROR 2: Overflow error of the viscoplastic strain rate evaluation: exponent too "
          "high: " +
          std::to_string(std::log(dt) + std::log(p) + e * (stress_ratio - 1.0)));
    }

    equiv_plastic_strain_rate = p * (std::exp(e * (stress_ratio - 1.0)) - 1.0);
  }

  return equiv_plastic_strain_rate;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<2, 1>
Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_derivatives_of_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const bool update_hist_var)
{
  // unwrap variables
  const double p = parameter()->strain_rate_pre_fac();         // prefactor of plastic strain rate
  const double e = 1.0 / parameter()->strain_rate_exp_fac();   // exponent of plastic strain rate
  const double sigma_Y0 = parameter()->init_yield_strength();  // initial yield strength
  const double B = parameter()->isotrop_harden_prefac();       // hardening prefactor
  const double N = parameter()->isotrop_harden_exp();          // hardening exponent

  // used equivalent plastic strain
  double used_equiv_plastic_strain = equiv_plastic_strain;

  // check whether the plastic strain is less than a set value (singularity in the derivatives
  // below)
  if (std::abs(equiv_plastic_strain) < 1.0e-16)
  {
    used_equiv_plastic_strain = 1.0e-16;
  }

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    throw std::runtime_error(
        "ERROR 1: Negative Plastic Strain " + std::to_string(equiv_plastic_strain));
  }

  // extraction of yield strength from the plastic strain and the material parameters
  const double yield_strength = sigma_Y0 + B * std::pow(used_equiv_plastic_strain, N);


  // computation of derivatives

  // first we set derivatives to 0
  Core::LinAlg::Matrix<2, 1> equiv_plastic_strain_rate_ders(true);

  // then we check the yield condition
  if (evaluate_stress_ratio(equiv_stress, used_equiv_plastic_strain) >= 1.0)
  {
    // compute the exact derivatives
    equiv_plastic_strain_rate_ders(0, 0) =
        p * std::exp(e * (equiv_stress / yield_strength - 1.0)) * e / yield_strength;

    equiv_plastic_strain_rate_ders(1, 0) = p * std::exp(e * (equiv_stress / yield_strength - 1.0)) *
                                           e * (-equiv_stress / std::pow(yield_strength, 2.0)) * B *
                                           N * std::pow(used_equiv_plastic_strain, N - 1.0);
  }

  return equiv_plastic_strain_rate_ders;
}

FOUR_C_NAMESPACE_CLOSE

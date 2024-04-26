/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of the isochoric contribution of an isotropic general power-type material in
terms of the first Cauchy-Green invariant

\level 1
*/
/*----------------------------------------------------------------------*/

#include "4C_matelast_iso1pow.hpp"

#include "4C_mat_par_material.hpp"

FOUR_C_NAMESPACE_OPEN


MAT::ELASTIC::PAR::Iso1Pow::Iso1Pow(const Teuchos::RCP<MAT::PAR::Material>& matdata)
    : Parameter(matdata), c_(*matdata->Get<double>("C")), d_(*matdata->Get<int>("D"))
{
}

MAT::ELASTIC::Iso1Pow::Iso1Pow(MAT::ELASTIC::PAR::Iso1Pow* params) : params_(params) {}

void MAT::ELASTIC::Iso1Pow::AddStrainEnergy(double& psi, const CORE::LINALG::Matrix<3, 1>& prinv,
    const CORE::LINALG::Matrix<3, 1>& modinv, const CORE::LINALG::Matrix<6, 1>& glstrain,
    const int gp, const int eleGID)
{
  // material Constants c and d
  const double c = params_->c_;
  const int d = params_->d_;

  // strain energy: Psi = C (\overline{I}_{\boldsymbol{C}}-3)^D
  // add to overall strain energy
  psi += c * pow((modinv(0) - 3.), d);
}

void MAT::ELASTIC::Iso1Pow::AddDerivativesModified(CORE::LINALG::Matrix<3, 1>& dPmodI,
    CORE::LINALG::Matrix<6, 1>& ddPmodII, const CORE::LINALG::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double c = params_->c_;
  const int d = params_->d_;

  if (d < 1)
    FOUR_C_THROW(
        "The Elast_Iso1Pow - material only works for positive integer exponents larger than one.");

  if (d == 1)
    dPmodI(0) += c * d;
  else
    dPmodI(0) += c * d * pow(modinv(0) - 3., d - 1.);

  if (d == 1)
    ddPmodII(0) += 0.;
  else if (d == 2)
    ddPmodII(0) += c * d * (d - 1.);
  else
    ddPmodII(0) += c * d * (d - 1.) * pow(modinv(0) - 3., d - 2.);
}
FOUR_C_NAMESPACE_CLOSE
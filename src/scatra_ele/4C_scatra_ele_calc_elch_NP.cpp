/*--------------------------------------------------------------------------*/
/*! \file

\brief evaluation of ScaTra elements for Nernst-Planck ion-transport equations

\level 2

*/
/*--------------------------------------------------------------------------*/
#include "4C_scatra_ele_calc_elch_NP.hpp"

#include "4C_discretization_fem_general_extract_values.hpp"
#include "4C_lib_discret.hpp"
#include "4C_mat_list.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_utils_elch.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>* DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::Instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = CORE::UTILS::MakeSingletonMap<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcElchNP<distype>>(
            new ScaTraEleCalcElchNP<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].Instance(
      CORE::UTILS::SingletonAction::create, numdofpernode, numscal, disname);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::ScaTraEleCalcElchNP(
    const int numdofpernode, const int numscal, const std::string& disname)
    : DRT::ELEMENTS::ScaTraEleCalcElch<distype>::ScaTraEleCalcElch(numdofpernode, numscal, disname),
      migrationstab_(true)
{
  // replace elch internal variable manager by internal variable manager for Nernst-Planck
  // formulation
  my::scatravarmanager_ = Teuchos::rcp(
      new ScaTraEleInternalVariableManagerElchNP<nsd_, nen_>(my::numscal_, myelch::elchparams_));

  return;
}


/*---------------------------------------------------------------------------------------*
 | calculate contributions to matrix and rhs (inside loop over all scalars)   fang 02/15 |
 *---------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatAndRhs(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to calculate
    CORE::LINALG::SerialDenseVector& erhs,  //!< element rhs to calculate+
    const int k,                            //!< index of current scalar
    const double fac,                       //!< domain-integration factor
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double rhsfac,      //!< time-integration factor for rhs times domain-integration factor
    const double taufac,      //!< tau times domain-integration factor
    const double timetaufac,  //!< domain-integration factor times tau times time-integration factor
    const double
        rhstaufac,  //!< time-integration factor for rhs times tau times domain-integration factor
    CORE::LINALG::Matrix<nen_, 1>&
        tauderpot,  //!< derivatives of stabilization parameter w.r.t. electric potential
    double& rhsint  //!< rhs at Gauss point
)
{
  // Compute residual of Nernst-Planck equation in strong form and subgrid-scale part of
  // concentration c_k
  const double residual = CalcRes(k, VarManager()->Phinp(k), VarManager()->Hist(k),
      VarManager()->ConvPhi(k), VarManager()->FRT(), VarManager()->MigConv(), rhsint);

  //--------------------------------------------------------------------------
  // 1) element matrix: instationary terms arising from Nernst-Planck equation
  //--------------------------------------------------------------------------

  if (not my::scatraparatimint_->IsStationary())
  {
    // 1a) element matrix: standard Galerkin mass term
    my::CalcMatMass(emat, k, fac, 1.);

    // 1b) element matrix: stabilization of mass term
    // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
    // available
  }

  //------------------------------------------------------------------------
  // 2) element matrix: stationary terms arising from Nernst-Planck equation
  //------------------------------------------------------------------------

  // 2a) element matrix: standard Galerkin convective term due to fluid flow
  my::CalcMatConv(emat, k, timefacfac, 1., VarManager()->SGConv());

  // 2b) element matrix: additional terms in conservative formulation if needed
  if (my::scatrapara_->IsConservative())
  {
    double vdiv(0.);
    my::GetDivergence(vdiv, my::evelnp_);
    my::CalcMatConvAddCons(emat, k, timefacfac, vdiv, 1.);
  }

  // 2c) element matrix: stabilization of convective term due to fluid flow and migration
  CalcMatConvStab(emat, k, timefacfac, taufac, timetaufac, tauderpot, VarManager()->FRT(),
      VarManager()->Conv(k), VarManager()->MigConv(), VarManager()->Phinp(k),
      VarManager()->GradPhi(k), residual);

  // 2d) element matrix: standard Galerkin diffusive term (constant diffusion coefficient)
  my::CalcMatDiff(emat, k, timefacfac);

  // 2e) element matrix: stabilization of diffusive term
  // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
  // available

  // 2f) element matrix: standard Galerkin migration term (can be split up into convective and
  // reactive parts)
  CalcMatMigr(
      emat, k, timefacfac, VarManager()->FRT(), VarManager()->MigConv(), VarManager()->Phinp(k));

  // 2g) element matrix: stabilization of reactive term due to migration
  // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
  // available

  //-------------------------------------------------------------------------------------------
  // 3) element matrix: stationary terms arising from governing equation for electric potential
  //-------------------------------------------------------------------------------------------

  // element matrix: standard Galerkin terms from governing equation for electric potential field
  switch (myelch::elchparams_
              ->EquPot())  // determine type of equation used for electric potential field
  {
    case INPAR::ELCH::equpot_enc:
    {
      myelch::CalcMatPotEquENC(emat, k, fac, my::scatraparatimint_->AlphaF());
      break;
    }
    case INPAR::ELCH::equpot_enc_pde:
    {
      CalcMatPotEquENCPDE(emat, k, timefacfac, VarManager()->FRT(), VarManager()->MigConv(),
          VarManager()->Phinp(k));
      break;
    }
    case INPAR::ELCH::equpot_enc_pde_elim:
    {
      CalcMatPotEquENCPDEElim(emat, k, timefacfac, VarManager()->FRT(), VarManager()->MigConv(),
          VarManager()->Phinp(k));
      break;
    }
    case INPAR::ELCH::equpot_poisson:
    {
      CalcMatPotEquPoisson(
          emat, k, fac, myelch::elchparams_->Epsilon(), myelch::elchparams_->Faraday());
      break;
    }
    case INPAR::ELCH::equpot_laplace:
    {
      // do nothing here, but later outside loop over scalars
      break;
    }
    default:
    {
      FOUR_C_THROW("Closing equation for electric potential not recognized!");
      break;
    }
  }

  //----------------------------------------------------------------------------
  // 4) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from Nernst-Planck equation
  //----------------------------------------------------------------------------

  // 4a) element rhs: standard Galerkin contributions from non-history part of instationary term if
  // needed
  if (not my::scatraparatimint_->IsStationary()) my::CalcRHSLinMass(erhs, k, rhsfac, fac, 1., 1.);

  // 4b) element rhs: standard Galerkin contributions from rhsint vector (contains body force vector
  // and history vector) need to adapt rhsint vector to time integration scheme first
  my::ComputeRhsInt(rhsint, 1., 1., VarManager()->Hist(k));
  my::CalcRHSHistAndSource(erhs, k, fac, rhsint);

  // 4c) element rhs: stabilization of mass term
  // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
  // available

  // 4d) element rhs: standard Galerkin convective term
  my::CalcRHSConv(erhs, k, rhsfac);

  // 4e) element rhs: additional terms in conservative formulation if needed
  if (my::scatrapara_->IsConservative())
  {
    double vdiv(0.);
    my::GetDivergence(vdiv, my::evelnp_);
    CalcRhsConvAddCons(erhs, k, rhsfac, VarManager()->Phinp(k), vdiv);
  }

  // 4f) element rhs: stabilization of convective term due to fluid flow and migration
  CalcRhsConvStab(erhs, k, rhstaufac, VarManager()->Conv(k), VarManager()->MigConv(), residual);

  // 4g) element rhs: standard Galerkin diffusion term
  my::CalcRHSDiff(erhs, k, rhsfac);

  // 4h) element rhs: stabilization of diffusive term
  // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
  // available

  // 4i) element rhs: standard Galerkin migration term (can be split up into convective and reactive
  // parts)
  CalcRhsMigr(erhs, k, rhsfac, VarManager()->MigConv(), VarManager()->Phinp(k));

  // 4j) element rhs: stabilization of reactive term due to migration
  // not implemented, only SUPG stabilization of convective term due to fluid flow and migration
  // available

  //----------------------------------------------------------------------------
  // 5) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from governing equation for electric potential
  //----------------------------------------------------------------------------

  // element rhs: standard Galerkin terms from governing equation for electric potential field
  switch (myelch::elchparams_
              ->EquPot())  // determine type of equation used for electric potential field
  {
    case INPAR::ELCH::equpot_enc:
    {
      myelch::CalcRhsPotEquENC(erhs, k, fac, VarManager()->Phinp(k));
      break;
    }
    case INPAR::ELCH::equpot_enc_pde:
    {
      CalcRhsPotEquENCPDE(erhs, k, rhsfac, VarManager()->MigConv(), VarManager()->Phinp(k),
          VarManager()->GradPhi(k));
      break;
    }
    case INPAR::ELCH::equpot_enc_pde_elim:
    {
      CalcRhsPotEquENCPDEElim(erhs, k, rhsfac, VarManager()->MigConv(), VarManager()->Phinp(k),
          VarManager()->GradPhi(k));
      break;
    }
    case INPAR::ELCH::equpot_poisson:
    {
      CalcRhsPotEquPoisson(erhs, k, fac, myelch::elchparams_->Epsilon(),
          myelch::elchparams_->Faraday(), VarManager()->Phinp(k), VarManager()->GradPot());
      break;
    }
    case INPAR::ELCH::equpot_laplace:
    {
      // do nothing here, but later outside loop over scalars
      break;
    }
    default:
    {
      FOUR_C_THROW("Closing equation for electric potential not recognized!");
      break;
    }
  }  // end switch(myelch::elchparams_->EquPot())
  return;
}


/*----------------------------------------------------------------------------------------*
 | calculate contributions to matrix and rhs (outside loop over all scalars)   fang 02/15 |
 *----------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatAndRhsOutsideScalarLoop(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to calculate
    CORE::LINALG::SerialDenseVector& erhs,  //!< element rhs to calculate
    const double fac,                       //!< domain-integration factor
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double rhsfac       //!< time-integration factor for rhs times domain-integration factor
)
{
  //-------------------------------------------------------------------------------------------
  // 3) element matrix: stationary terms arising from governing equation for electric potential
  //-------------------------------------------------------------------------------------------

  // element matrix: standard Galerkin terms from governing equation for electric potential field
  switch (myelch::elchparams_
              ->EquPot())  // determine type of equation used for electric potential field
  {
    case INPAR::ELCH::equpot_enc:
    case INPAR::ELCH::equpot_enc_pde:
    case INPAR::ELCH::equpot_enc_pde_elim:
    case INPAR::ELCH::equpot_poisson:
    {
      // has already been evaluated inside loop over scalars
      break;
    }
    case INPAR::ELCH::equpot_laplace:
    {
      CalcMatPotEquLaplace(emat, fac);
      break;
    }
    default:
    {
      FOUR_C_THROW("Closing equation for electric potential not recognized!");
      break;
    }
  }  // end switch(myelch::elchparams_->EquPot())

  //----------------------------------------------------------------------------
  // 5) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from governing equation for electric potential
  //----------------------------------------------------------------------------

  // element rhs: standard Galerkin terms from governing equation for electric potential field
  switch (myelch::elchparams_
              ->EquPot())  // determine type of equation used for electric potential field
  {
    case INPAR::ELCH::equpot_enc:
    case INPAR::ELCH::equpot_enc_pde:
    case INPAR::ELCH::equpot_enc_pde_elim:
    case INPAR::ELCH::equpot_poisson:
    {
      // has already been evaluated inside loop over scalars
      break;
    }
    case INPAR::ELCH::equpot_laplace:
    {
      CalcRhsPotEquLaplace(erhs, fac, VarManager()->GradPot());
      break;
    }
    default:
    {
      FOUR_C_THROW("Closing equation for electric potential not recognized!");
      break;
    }
  }  // end switch(myelch::elchparams_->EquPot())
  return;
}


/*-----------------------------------------------------------------------------------------*
 | CalcRes: Residual of Nernst-Planck equation in strong form (private)         fang 06/14 |
 *------------------------------------------------------ ----------------------------------*/
template <CORE::FE::CellType distype>
double DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRes(
    const int k,           //!< index of current scalar
    const double conint,   //!< concentration at GP
    const double hist,     //!< history value at GP
    const double convphi,  //!< convective term (without convective part of migration term)
    const double frt,      //!< F/(RT)
    const CORE::LINALG::Matrix<nen_, 1>&
        migconv,  //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
    const double
        rhsint  //!< rhs of Nernst-Planck equation (not of Newton-Raphson scheme) at Gauss point
)
{
  // Compute convective term including convective part of migration term
  const double convmigphi = convphi + myelch::DiffManager()->GetIsotropicDiff(k) *
                                          myelch::DiffManager()->GetValence(k) *
                                          (migconv.Dot(my::ephinp_[k]));

  // Compute diffusive term and reactive part of migration term (only significant for higher-order
  // elements)
  double diffphi(0.);
  double reamigphi(0.);

  if (my::use2ndderiv_)
  {
    CORE::LINALG::Matrix<nen_, 1> laplace(true);
    my::GetLaplacianStrongForm(laplace);

    diffphi = myelch::DiffManager()->GetIsotropicDiff(k) * laplace.Dot(my::ephinp_[k]);
    reamigphi = -frt * myelch::DiffManager()->GetIsotropicDiff(k) *
                myelch::DiffManager()->GetValence(k) * laplace.Dot(my::ephinp_[my::numscal_]) *
                conint;
  }

  if (my::scatraparatimint_->IsStationary())
    return convmigphi - diffphi + reamigphi - rhsint;
  else if (my::scatraparatimint_->IsGenAlpha())
    return hist + convmigphi - diffphi + reamigphi - rhsint;
  else
    return conint - hist +
           my::scatraparatimint_->TimeFac() * (convmigphi - diffphi + reamigphi - rhsint);
}  // ScaTraEleCalcElchNP<distype>::CalcRes


/*-------------------------------------------------------------------------------------------------------*
 | CalcMat: SUPG Stabilization of convective term due to fluid flow and migration (private)   fang
 06/14 |
 *------------------------------------------------------
 ------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatConvStab(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to calculate
    const int k,                            //!< index of current scalar
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double taufac,      //!< stabilization parameter tau times domain-integration factor
    const double timetaufac,  //!< domain-integration factor times tau times time-integration factor
    CORE::LINALG::Matrix<nen_, 1>&
        tauderpot,     //!< derivatives of stabilization parameter w.r.t. electric potential
    const double frt,  //!< F/(RT)
    const CORE::LINALG::Matrix<nen_, 1>&
        conv,  //!< convection operator: u_x*N,x + u_y*N,y + u_z*N,z
    const CORE::LINALG::Matrix<nen_, 1>&
        migconv,          //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
    const double conint,  //!< concentration at GP
    const CORE::LINALG::Matrix<nsd_, 1>& gradphi,  //!< gradient of concentration at GP
    const double residual  //!< residual of Nernst-Planck equation in strong form
)
{
  // Compute Laplacian N,xx  +  N,yy +  N,zz of all shape functions at current integration point if
  // needed
  CORE::LINALG::Matrix<nen_, 1> laplace(true);
  if (my::use2ndderiv_) my::GetLaplacianStrongForm(laplace);

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    // compute effective convective stabilization operator
    double conv_eff_vi = conv(vi);
    if (migrationstab_)
      conv_eff_vi += myelch::DiffManager()->GetIsotropicDiff(k) *
                     myelch::DiffManager()->GetValence(k) * migconv(vi);

    // shortcuts
    const double timetaufac_conv_eff_vi = timetaufac * conv_eff_vi;
    const double timetaufac_conv_eff_vi_conint_k_frt_valence_k =
        timetaufac_conv_eff_vi * conint * frt * myelch::DiffManager()->GetValence(k);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      // matrix entries
      double matvalconc = 0.;
      double matvalpot = 0.;

      // 1) transient term
      if (not my::scatraparatimint_->IsStationary())
        matvalconc += taufac * conv_eff_vi * my::funct_(ui);

      // 2) convective term due to fluid flow and migration
      // 2a) linearization of residual w.r.t. concentration c_k
      matvalconc += timetaufac * conv_eff_vi *
                    (conv(ui) + myelch::DiffManager()->GetIsotropicDiff(k) *
                                    myelch::DiffManager()->GetValence(k) * migconv(ui));

      // 2b) linearization of residual w.r.t. electric potential Phi
      double laplawf(0.);
      my::GetLaplacianWeakFormRHS(laplawf, gradphi, ui);
      matvalpot -= timetaufac * conv_eff_vi * myelch::DiffManager()->GetIsotropicDiff(k) *
                   myelch::DiffManager()->GetValence(k) * frt * laplawf;

      if (migrationstab_)
      {
        // 2c) linearization of migration operator w.r.t. concentration c_k
        // not necessary, since migration operator not a function of c_k

        // 2d) linearization of migration operator w.r.t. electric potential Phi
        double laplacewf(0.);
        my::GetLaplacianWeakForm(laplacewf, ui, vi);
        matvalpot -= timetaufac * residual * myelch::DiffManager()->GetIsotropicDiff(k) *
                     myelch::DiffManager()->GetValence(k) * frt * laplacewf;
      }

      if (not SCATRA::IsBinaryElectrolyte(myelch::DiffManager()->GetValence()))
      {
        // 2e) linearization of tau w.r.t. concentration c_k
        // not necessary, since tau not a function of c_k

        // 2f) linearization of tau w.r.t. electric potential Phi (only non-zero for
        // Taylor_Hughes_Zarins at the moment)
        matvalpot += timefacfac * tauderpot(ui) * conv_eff_vi * residual;
      }

      if (my::use2ndderiv_)
      {
        // 3) diffusive term
        // 3a) linearization w.r.t. concentration c_k
        matvalconc -=
            timetaufac_conv_eff_vi * myelch::DiffManager()->GetIsotropicDiff(k) * laplace(ui);

        // 3b) linearization w.r.t. electric potential Phi
        // not necessary, since diffusive term not a function of Phi

        // 4) reactive term due to migration
        // 4a) linearization w.r.t. concentration c_k
        matvalconc -= timetaufac_conv_eff_vi * frt * myelch::DiffManager()->GetIsotropicDiff(k) *
                      myelch::DiffManager()->GetValence(k) *
                      laplace.Dot(my::ephinp_[my::numscal_]) * my::funct_(ui);

        // 4b) linearization w.r.t. electric potential Phi
        matvalpot -= timetaufac_conv_eff_vi_conint_k_frt_valence_k *
                     myelch::DiffManager()->GetIsotropicDiff(k) * laplace(ui);
      }

      // try to access the element matrix not too often, can be costly
      const int fvi = vi * my::numdofpernode_ + k;
      emat(fvi, ui * my::numdofpernode_ + k) += matvalconc;
      emat(fvi, ui * my::numdofpernode_ + my::numscal_) += matvalpot;
    }
  }

  return;
}  // ScaTraEleCalcElchNP<distype>::CalcMatMassStab


/*-----------------------------------------------------------------------*
 |  CalcMat: Migration term (private)                         fang 05/14 |
 *-------------------------------------------------- --------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatMigr(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to calculate
    const int k,                            //!< index of current scalar
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double frt,         //!< F/(RT)
    const CORE::LINALG::Matrix<nen_, 1>&
        migconv,         //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
    const double conint  //!< concentration at GP
)
{
  const double timefacfac_diffus_valence_k = timefacfac *
                                             myelch::DiffManager()->GetIsotropicDiff(k) *
                                             myelch::DiffManager()->GetValence(k);
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = timefacfac_diffus_valence_k * migconv(vi);
    const int fvi = vi * my::numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * my::numdofpernode_ + k;

      // a) derivative w.r.t. concentration c_k
      emat(fvi, fui) -= v * my::funct_(ui);

      // b) derivative w.r.t. electric potential
      double laplawf(0.);
      my::GetLaplacianWeakForm(laplawf, ui, vi);
      emat(fvi, ui * my::numdofpernode_ + my::numscal_) +=
          frt * timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) *
          myelch::DiffManager()->GetValence(k) * conint * laplawf;
    }
  }

  return;
}  // ScaTraEleCalcElchNP<distype>::CalcMatMigr


/*-----------------------------------------------------------------------*
 |  CalcMat: Electroneutrality in PDE form (private)          fang 05/14 |
 *-----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatPotEquENCPDE(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double frt,         //!< F/(RT)
    const CORE::LINALG::Matrix<nen_, 1>& migconv,  //!< migration operator
    const double conint                            //!< concentration at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    // Inclusion of time integration factor results in a matrix with better condition number
    const double timefacfac_diffus_valence_k_mig_vi =
        timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) *
        myelch::DiffManager()->GetValence(k) * migconv(vi);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * my::numdofpernode_ + k;

      double laplawf(0.);
      my::GetLaplacianWeakForm(laplawf, ui, vi);

      // use 2nd order pde derived from electroneutrality condition (k=1,...,m)
      // a) derivative w.r.t. concentration c_k
      emat(pvi, fui) -= myelch::DiffManager()->GetValence(k) *
                        (timefacfac_diffus_valence_k_mig_vi * my::funct_(ui));
      emat(pvi, fui) += myelch::DiffManager()->GetValence(k) *
                        (timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) * laplawf);
      // b) derivative w.r.t. electric potential
      emat(pvi, ui * my::numdofpernode_ + my::numscal_) +=
          myelch::DiffManager()->GetValence(k) *
          (frt * timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) *
              myelch::DiffManager()->GetValence(k) * conint * laplawf);
    }  // for ui
  }    // for vi

  return;
}


/*-------------------------------------------------------------------------------------------*
 |  CalcMat: ENC in PDE form with NP equation for species m eliminated (private)  fang 05/14 |
 *-------------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatPotEquENCPDEElim(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double frt,         //!< F/(RT)
    const CORE::LINALG::Matrix<nen_, 1>& migconv,  //!< migration operator
    const double conint                            //!< concentration at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    // Inclusion of time integration factor results in a matrix with better condition number
    const double timefacfac_diffus_valence_k_mig_vi =
        timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) *
        myelch::DiffManager()->GetValence(k) * migconv(vi);
    const double timefacfac_diffus_valence_m_mig_vi =
        timefacfac * myelch::DiffManager()->GetIsotropicDiff(my::numscal_) *
        myelch::DiffManager()->GetValence(my::numscal_) * migconv(vi);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      // matrix entries
      double matvalconc = 0.;
      double matvalpot = 0.;

      double laplawf(0.);
      my::GetLaplacianWeakForm(laplawf, ui, vi);

      // use 2nd order pde derived from electroneutrality condition (k=1,...,m-1)
      // a) derivative w.r.t. concentration c_k
      matvalconc -= timefacfac_diffus_valence_k_mig_vi * my::funct_(ui);
      matvalconc += timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) * laplawf;
      // b) derivative w.r.t. electric potential
      matvalpot += frt * timefacfac * myelch::DiffManager()->GetIsotropicDiff(k) *
                   myelch::DiffManager()->GetValence(k) * conint * laplawf;

      // care for eliminated species with index m
      // Note: diffus_ and valence_ vectors were extended in GetMaterialParams() so that they
      // also contain the properties of the eliminated species at index m (= my::numscal_))
      // a) derivative w.r.t. concentration c_k
      matvalconc += timefacfac_diffus_valence_m_mig_vi * my::funct_(ui);
      matvalconc -= timefacfac * myelch::DiffManager()->GetIsotropicDiff(my::numscal_) * laplawf;
      // b) derivative w.r.t. electric potential
      matvalpot -= frt * timefacfac * myelch::DiffManager()->GetIsotropicDiff(my::numscal_) *
                   myelch::DiffManager()->GetValence(my::numscal_) * conint * laplawf;

      // try to access the element matrix not too often, can be costly
      const int fui = ui * my::numdofpernode_ + k;
      emat(pvi, fui) += myelch::DiffManager()->GetValence(k) * matvalconc;
      const int pui = ui * my::numdofpernode_ + my::numscal_;
      emat(pvi, pui) += myelch::DiffManager()->GetValence(k) * matvalpot;
    }  // for ui
  }    // for vi

  return;
}


/*-------------------------------------------------------------------------------------*
 |  CalcMat: Poisson equation for electric potential (private)              fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatPotEquPoisson(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const double fac,                       //!< domain-integration factor
    const double epsilon,                   //!< dielectric constant
    const double faraday                    //!< Faraday constant
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;
    const double alphaF_valence_k_fac_funct_vi = my::scatraparatimint_->AlphaF() *
                                                 myelch::DiffManager()->GetValence(k) * fac *
                                                 my::funct_(vi);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      // We have a loop over the species index k around. So prevent that the potential term is added
      // more than once!
      if (k == 0)
      {
        const int pui = ui * my::numdofpernode_ + my::numscal_;
        double laplawf(0.);
        my::GetLaplacianWeakForm(laplawf, ui, vi);

        const double epsbyF = epsilon / faraday;

        emat(pvi, pui) += my::scatraparatimint_->AlphaF() * fac * epsbyF * laplawf;
      }

      const int fui = ui * my::numdofpernode_ + k;

      // electroneutrality condition (only derivative w.r.t. concentration c_k)
      emat(pvi, fui) -= alphaF_valence_k_fac_funct_vi * my::funct_(ui);
    }  // for ui
  }    // for vi

  return;
}


/*-------------------------------------------------------------------------------------*
 |  CalcMat: Laplace equation for electric potential (private)              fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcMatPotEquLaplace(
    CORE::LINALG::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const double fac                        //!< domain-integration factor
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int pui = ui * my::numdofpernode_ + my::numscal_;

      double laplawf(0.);
      my::GetLaplacianWeakForm(laplawf, ui, vi);

      emat(pvi, pui) += my::scatraparatimint_->AlphaF() * fac * laplawf;
    }  // for ui
  }    // for vi

  return;
}


/*-----------------------------------------------------------------------------------------*
 |  CalcRhs: Additional contributions from conservative formulation (private)   fang 05/14 |
 *-----------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsConvAddCons(
    CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const double conint,  //!< concentration at GP
    const double vdiv     //!< velocity divergence
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
    erhs[vi * my::numdofpernode_ + k] -= rhsfac * my::funct_(vi) * conint * vdiv;

  return;
}


/*-------------------------------------------------------------------------------------------------------*
 | CalcRhs: SUPG Stabilization of convective term due to fluid flow and migration (private)   fang
 06/14 |
 *------------------------------------------------------
 ------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsConvStab(
    CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double
        rhstaufac,  //!< time-integration factor for rhs times tau times domain-integration factor
    const CORE::LINALG::Matrix<nen_, 1>&
        conv,  //!< convection operator: u_x*N,x + u_y*N,y + u_z*N,z
    const CORE::LINALG::Matrix<nen_, 1>&
        migconv,           //!< migration operator: -F/(RT) \grad{\Phi} * \grad{N}
    const double residual  //!< residual of Nernst-Planck equation in strong form
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    erhs[fvi] -= rhstaufac * conv(vi) * residual;

    if (migrationstab_)
      erhs[fvi] -= rhstaufac * myelch::DiffManager()->GetIsotropicDiff(k) *
                   myelch::DiffManager()->GetValence(k) * migconv(vi) * residual;
  }

  return;
}  // ScaTraEleCalcElchNP<distype>::CalcRhsConvStab


/*-------------------------------------------------------------------------------------*
 |  CalcRhs: Migration term (private)                                       fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsMigr(
    CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const CORE::LINALG::Matrix<nen_, 1>& migconv,  //!< migration operator
    const double conint                            //!< concentration at GP
)
{
  const double rhsfac_con_diffus_valence_k = rhsfac * conint *
                                             myelch::DiffManager()->GetIsotropicDiff(k) *
                                             myelch::DiffManager()->GetValence(k);

  for (unsigned vi = 0; vi < nen_; ++vi)
    erhs[vi * my::numdofpernode_ + k] += rhsfac_con_diffus_valence_k * migconv(vi);

  return;
}


/*-------------------------------------------------------------------------------------*
 |  CalcRhs: Electroneutrality condition in PDE form (private)              fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsPotEquENCPDE(
    CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const CORE::LINALG::Matrix<nen_, 1>& migconv,  //!< migration operator
    const double conint,                           //!< concentration at GP
    const CORE::LINALG::Matrix<nsd_, 1>& gradphi   //!< gradient of concentration at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    double laplawf(0.);
    my::GetLaplacianWeakFormRHS(laplawf, gradphi, vi);

    // use 2nd order pde derived from electroneutrality condition (k=1,...,m)
    // Inclusion of time integration factor results in a matrix with better condition number
    erhs[vi * my::numdofpernode_ + my::numscal_] +=
        rhsfac * myelch::DiffManager()->GetValence(k) *
        (myelch::DiffManager()->GetIsotropicDiff(k) * myelch::DiffManager()->GetValence(k) *
                conint * migconv(vi) -
            myelch::DiffManager()->GetIsotropicDiff(k) * laplawf);
  }  // for vi

  return;
}


/*-------------------------------------------------------------------------------------------*
 |  CalcRhs: ENC in PDE form with NP equation for species m eliminated (private)  fang 05/14 |
 *-------------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsPotEquENCPDEElim(
    CORE::LINALG::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const CORE::LINALG::Matrix<nen_, 1>& migconv,  //!< migration operator
    const double conint,                           //!< concentration at GP
    const CORE::LINALG::Matrix<nsd_, 1>& gradphi   //!< gradient of concentration at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    double laplawf(0.);
    my::GetLaplacianWeakFormRHS(laplawf, gradphi, vi);

    // use 2nd order pde derived from electroneutrality condition (k=0,...,m-1)
    // Inclusion of time integration factor results in a matrix with better condition number
    erhs[pvi] += rhsfac * myelch::DiffManager()->GetValence(k) *
                 (myelch::DiffManager()->GetIsotropicDiff(k) *
                         myelch::DiffManager()->GetValence(k) * conint * migconv(vi) -
                     myelch::DiffManager()->GetIsotropicDiff(k) * laplawf);

    // care for eliminated species with index m
    // Note: diffus_ and valence_ vectors were extended in GetMaterialParams() so that they
    // also contain the properties of the eliminated species at index m (= my::numscal_))
    erhs[pvi] -= rhsfac * myelch::DiffManager()->GetValence(k) *
                 (myelch::DiffManager()->GetIsotropicDiff(my::numscal_) *
                         myelch::DiffManager()->GetValence(my::numscal_) * conint * migconv(vi) -
                     myelch::DiffManager()->GetIsotropicDiff(my::numscal_) * laplawf);
  }  // for vi

  return;
}


/*-------------------------------------------------------------------------------------*
 |  CalcRhs: Poisson equation for electric potential (private)              fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsPotEquPoisson(
    CORE::LINALG::SerialDenseVector& erhs,        //!< element vector to be filled
    const int k,                                  //!< index of current scalar
    const double fac,                             //!< domain-integration factor
    const double epsilon,                         //!< dielectric constant
    const double faraday,                         //!< Faraday constant
    const double conint,                          //!< concentration at GP
    const CORE::LINALG::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    // We have a loop over the species index k around. So prevent that the potential term is added
    // more than once!
    if (k == 0)
    {
      double laplawf(0.);
      my::GetLaplacianWeakFormRHS(laplawf, gradpot, vi);

      const double epsbyF = epsilon / faraday;

      erhs[pvi] -= fac * epsbyF * laplawf;
    }

    // residuum of Poisson equation on the rhs
    erhs[pvi] += myelch::DiffManager()->GetValence(k) * fac * my::funct_(vi) * conint;
  }  // for vi

  return;
}


/*-------------------------------------------------------------------------------------*
 |  CalcRhs: Laplace equation for electric potential (private)              fang 05/14 |
 *-------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcRhsPotEquLaplace(
    CORE::LINALG::SerialDenseVector& erhs,        //!< element vector to be filled
    const double fac,                             //!< domain-integration factor
    const CORE::LINALG::Matrix<nsd_, 1>& gradpot  //!< gradient of potential at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int pvi = vi * my::numdofpernode_ + my::numscal_;

    double laplawf(0.);
    my::GetLaplacianWeakFormRHS(laplawf, gradpot, vi);

    erhs[pvi] -= fac * laplawf;
  }  // for vi

  return;
}


/*------------------------------------------------------------------------*
 |  Correct sysmat for fluxes across DC                        fang 05/14 |
 *------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CorrectionForFluxAcrossDC(
    DRT::Discretization& discretization, const std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& emat, CORE::LINALG::SerialDenseVector& erhs)
{
  if ((myelch::elchparams_->EquPot() == INPAR::ELCH::equpot_enc_pde) or
      (myelch::elchparams_->EquPot() == INPAR::ELCH::equpot_enc_pde_elim))
  {
    // get dirichlet toggle from the discretization
    Teuchos::RCP<const Epetra_Vector> dctoggle = discretization.GetState("dctoggle");
    std::vector<double> mydctoggle(lm.size());
    CORE::FE::ExtractMyValues(*dctoggle, mydctoggle, lm);

    double val = 0.;
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      for (int k = 0; k < my::numscal_; ++k)
      {
        //
        if (mydctoggle[vi * my::numdofpernode_ + k] == 1)
        {
          // std::cout<<"Ele Id = "<<ele->Id()<<"  Found one Dirichlet node for vi="<<vi<<std::endl;
          // std::cout<<"Dirichlet is on for k="<<k<<std::endl;
          // std::cout<<"k="<<k<<"  val="<<val<<" valence_k="<<valence_[k]<<std::endl;

          const int fvi = vi * my::numdofpernode_ + k;

          // We use the fact, that the rhs vector value for boundary nodes
          // is equivalent to the integrated negative normal flux
          // due to diffusion and migration
          val = erhs[fvi];
          erhs[vi * my::numdofpernode_ + my::numscal_] +=
              myelch::DiffManager()->GetValence(k) * (-val);

          // corresponding linearization
          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            val = emat(vi * my::numdofpernode_ + k, ui * my::numdofpernode_ + k);
            emat(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + k) +=
                myelch::DiffManager()->GetValence(k) * (-val);
            val = emat(vi * my::numdofpernode_ + k, ui * my::numdofpernode_ + my::numscal_);
            emat(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + my::numscal_) +=
                myelch::DiffManager()->GetValence(k) * (-val);
          }
        }  // if mydctoggle
      }    // for k
    }      // for vi
  }        // if EquPot()

  return;
}


/*----------------------------------------------------------------------*
 |  get the material constants  (private)                     ehrl 01/14|
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::GetMaterialParams(
    const DRT::Element* ele,      //!< the element we are dealing with
    std::vector<double>& densn,   //!< density at t_(n)
    std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,  //!< density at t_(n+alpha_M)
    double& visc,                 //!< fluid viscosity
    const int iquad               //!< id of current gauss point (default = -1)
)
{
  // get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

  if (material->MaterialType() == INPAR::MAT::m_matlist)
  {
    const Teuchos::RCP<const MAT::MatList>& actmat =
        Teuchos::rcp_dynamic_cast<const MAT::MatList>(material);
    if (actmat->NumMat() < my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP<MAT::Material> singlemat = actmat->MaterialById(matid);

      Materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
    FOUR_C_THROW("Invalid material type!");

  return;
}  // ScaTraEleCalc::GetMaterialParams


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    ehrl 11/13 |
 *----------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::Materials(
    const Teuchos::RCP<const MAT::Material> material,  //!< pointer to current material
    const int k,                                       //!< id of current scalar
    double& densn,                                     //!< density at t_(n)
    double& densnp,                                    //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,                                    //!< density at t_(n+alpha_M)
    double& visc,                                      //!< fluid viscosity
    const int iquad                                    //!< id of current gauss point
)
{
  if (material->MaterialType() == INPAR::MAT::m_ion)
    myelch::utils_->MatIon(material, k, myelch::elchparams_->EquPot(), myelch::DiffManager());
  else
    FOUR_C_THROW("Material type is not supported");

  return;
}


/*--------------------------------------------------------------------------*
 | Calculate quantities used for stabilization (protected)       fang 06/14 |
 *--------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::PrepareStabilization(
    std::vector<double>& tau,  //!< stabilization parameters (one per transported scalar)
    std::vector<CORE::LINALG::Matrix<nen_, 1>>&
        tauderpot,  //!< derivatives of stabilization parameters w.r.t. electric potential
    const std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_f)
    const double vol                    //!< element volume
)
{
  // special stabilization parameters in case of binary electrolyte
  if (SCATRA::IsBinaryElectrolyte(myelch::DiffManager()->GetValence()))
  {
    // do not include migration operator in stabilization terms
    migrationstab_ = false;

    // use effective diffusion coefficient for binary electrolyte solutions
    double resdiffus = SCATRA::CalResDiffCoeff(myelch::DiffManager()->GetValence(),
        myelch::DiffManager()->GetIsotropicDiff(),
        SCATRA::GetIndicesBinaryElectrolyte(myelch::DiffManager()->GetValence()));

    // loop over transported scalars
    for (int k = 0; k < my::numscal_; ++k)
    {
      // calculate stabilization parameter tau for charged species
      if (abs(myelch::DiffManager()->GetValence(k)) > 1.e-10)
        my::CalcTau(tau[k], resdiffus,
            my::reamanager_->GetStabilizationCoeff(k, my::scatravarmanager_->Phinp(k)), densnp[k],
            VarManager()->ConVel(k), vol);
      else
        // calculate stabilization parameter tau for uncharged species
        my::CalcTau(tau[k], myelch::DiffManager()->GetIsotropicDiff(k),
            my::reamanager_->GetStabilizationCoeff(k, my::scatravarmanager_->Phinp(k)), densnp[k],
            VarManager()->ConVel(k), vol);
    }
  }

  else
  {
    // only include migration operator in stabilization terms in case tau is computed according to
    // Taylor, Hughes, and Zarins
    switch (my::scatrapara_->TauDef())
    {
      case INPAR::SCATRA::tau_taylor_hughes_zarins:
      case INPAR::SCATRA::tau_taylor_hughes_zarins_wo_dt:
      {
        migrationstab_ = true;
        break;
      }
      default:
      {
        migrationstab_ = false;
        break;
      }
    }

    // loop over transported scalars
    for (int k = 0; k < my::numscal_; ++k)
    {
      // Compute effective velocity as sum of convective and migration velocities
      CORE::LINALG::Matrix<nsd_, 1> veleff(VarManager()->ConVel(k));
      veleff.Update(
          myelch::DiffManager()->GetValence(k) * myelch::DiffManager()->GetIsotropicDiff(k),
          VarManager()->MigVelInt(), 1.);

      // calculate stabilization parameter tau
      my::CalcTau(tau[k], myelch::DiffManager()->GetIsotropicDiff(k),
          my::reamanager_->GetStabilizationCoeff(k, my::scatravarmanager_->Phinp(k)), densnp[k],
          veleff, vol);

      switch (my::scatrapara_->TauDef())
      {
        case INPAR::SCATRA::tau_taylor_hughes_zarins:
        case INPAR::SCATRA::tau_taylor_hughes_zarins_wo_dt:
        {
          // Calculate derivative of tau w.r.t. electric potential
          CalcTauDerPotTaylorHughesZarins(tauderpot[k], tau[k], densnp[k], VarManager()->FRT(),
              myelch::DiffManager()->GetIsotropicDiff(k) * myelch::DiffManager()->GetValence(k),
              veleff);
          break;
        }
        default:
        {
          break;
        }
      }
    }
  }

  return;
}  // ScaTraEleCalcElch<distype>::PrepareStabilization


/*-----------------------------------------------------------------------------------------------------------------------*
 | Calculate derivative of tau w.r.t. electric potential according to Taylor, Hughes and Zarins
 (protected)   fang 06/14 |
 *-----------------------------------------------------------------------------------------------------------------------*/
template <CORE::FE::CellType distype>
void DRT::ELEMENTS::ScaTraEleCalcElchNP<distype>::CalcTauDerPotTaylorHughesZarins(
    CORE::LINALG::Matrix<nen_, 1>&
        tauderpot,        //!< derivatives of stabilization parameter w.r.t. electric potential
    double& tau,          //!< stabilization parameter
    const double densnp,  //!< density at t_(n+1)
    const double frt,     //!< F/(RT)
    const double diffusvalence,                  //!< diffusion coefficient times valence
    const CORE::LINALG::Matrix<nsd_, 1>& veleff  //!< effective convective velocity (fluid velocity
                                                 //!< plus migration velocity if applicable)
)
{
  /*
  literature:
  1) C.A. Taylor, T.J.R. Hughes, C.K. Zarins, Finite element modeling
  of blood flow in arteries, Comput. Methods Appl. Mech. Engrg. 158
  (1998) 155-196.
  2) V. Gravemeier, W.A. Wall, An algebraic variational multiscale-
  multigrid method for large-eddy simulation of turbulent variable-
  density flow at low Mach number, J. Comput. Phys. 229 (2010)
  6047-6070.
  */

  // Initialization
  tauderpot.Clear();

  // Compute entries of covariant metric tensor
  double G;
  const double dens_sqr = densnp * densnp;
  for (unsigned nn = 0; nn < nsd_; ++nn)
  {
    for (unsigned rr = 0; rr < nsd_; ++rr)
    {
      G = my::xij_(nn, 0) * my::xij_(rr, 0);

      for (unsigned tt = 1; tt < nsd_; ++tt) G += my::xij_(nn, tt) * my::xij_(rr, tt);

      for (unsigned jj = 0; jj < nen_; ++jj)
      {
        tauderpot(jj) +=
            dens_sqr * frt * diffusvalence *
            ((my::derxy_(nn, jj) * G * veleff(rr, 0)) + (veleff(nn, 0) * G * my::derxy_(rr, jj)));
      }
    }
  }

  // Finalize derivative of present tau w.r.t electric potential
  // Note: Factor alpha_f in gen-alpha time integration scheme is included at a later point
  tauderpot.Scale(0.5 * tau * tau * tau);

  return;
}


// template classes

// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::line2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::line3>;

// 2D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::tri3>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::tri6>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::quad4>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::quad9>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::nurbs9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::hex8>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::hex27>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::tet4>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::tet10>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::pyramid5>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchNP<CORE::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
/*---------------------------------------------------------------------*/
/*! \file
\brief A class to perform integrations of nitsche related terms

\level 3


*/
/*---------------------------------------------------------------------*/
#include "baci_contact_nitsche_integrator.H"

#include "baci_contact_element.H"
#include "baci_contact_nitsche_utils.H"
#include "baci_contact_node.H"
#include "baci_contact_paramsinterface.H"
#include "baci_discretization_fem_general_utils_boundary_integration.H"
#include "baci_mat_elasthyper.H"
#include "baci_so3_base.H"

#include <Epetra_FEVector.h>

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::CoIntegratorNitsche::IntegrateGP_3D(MORTAR::MortarElement& sele,
    MORTAR::MortarElement& mele, CORE::LINALG::SerialDenseVector& sval,
    CORE::LINALG::SerialDenseVector& lmval, CORE::LINALG::SerialDenseVector& mval,
    CORE::LINALG::SerialDenseMatrix& sderiv, CORE::LINALG::SerialDenseMatrix& mderiv,
    CORE::LINALG::SerialDenseMatrix& lmderiv,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, CORE::GEN::pairedvector<int, double>& derivjac, double* normal,
    std::vector<CORE::GEN::pairedvector<int, double>>& dnmap_unit, double& gap,
    CORE::GEN::pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<CORE::GEN::pairedvector<int, double>>& derivsxi,
    std::vector<CORE::GEN::pairedvector<int, double>>& derivmxi)
{
  GPTSForces<3>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt, gap,
      deriv_gap, normal, dnmap_unit, sxi, mxi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::CoIntegratorNitsche::IntegrateGP_2D(MORTAR::MortarElement& sele,
    MORTAR::MortarElement& mele, CORE::LINALG::SerialDenseVector& sval,
    CORE::LINALG::SerialDenseVector& lmval, CORE::LINALG::SerialDenseVector& mval,
    CORE::LINALG::SerialDenseMatrix& sderiv, CORE::LINALG::SerialDenseMatrix& mderiv,
    CORE::LINALG::SerialDenseMatrix& lmderiv,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, CORE::GEN::pairedvector<int, double>& derivjac, double* normal,
    std::vector<CORE::GEN::pairedvector<int, double>>& dnmap_unit, double& gap,
    CORE::GEN::pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<CORE::GEN::pairedvector<int, double>>& derivsxi,
    std::vector<CORE::GEN::pairedvector<int, double>>& derivmxi)
{
  GPTSForces<2>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt, gap,
      deriv_gap, normal, dnmap_unit, sxi, mxi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::CoIntegratorNitsche::GPTSForces(MORTAR::MortarElement& sele,
    MORTAR::MortarElement& mele, const CORE::LINALG::SerialDenseVector& sval,
    const CORE::LINALG::SerialDenseMatrix& sderiv,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dsxi,
    const CORE::LINALG::SerialDenseVector& mval, const CORE::LINALG::SerialDenseMatrix& mderiv,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dmxi, const double jac,
    const CORE::GEN::pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const CORE::GEN::pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<CORE::GEN::pairedvector<int, double>>& deriv_contact_normal, double* sxi,
    double* mxi)
{
  if (sele.Owner() != Comm_.MyPID()) return;

  if (dim != Dim()) dserror("dimension inconsistency");

  if (frtype_ != INPAR::CONTACT::friction_none && dim != 3) dserror("only 3D friction");
  if (frtype_ != INPAR::CONTACT::friction_none && frtype_ != INPAR::CONTACT::friction_coulomb &&
      frtype_ != INPAR::CONTACT::friction_tresca)
    dserror("only coulomb or tresca friction");
  if (frtype_ == INPAR::CONTACT::friction_coulomb && frcoeff_ < 0.)
    dserror("negative coulomb friction coefficient");
  if (frtype_ == INPAR::CONTACT::friction_tresca && frbound_ < 0.)
    dserror("negative tresca friction bound");

  CORE::LINALG::Matrix<dim, 1> slave_normal, master_normal;
  std::vector<CORE::GEN::pairedvector<int, double>> deriv_slave_normal(0, 0);
  std::vector<CORE::GEN::pairedvector<int, double>> deriv_master_normal(0, 0);
  sele.ComputeUnitNormalAtXi(sxi, slave_normal.A());
  mele.ComputeUnitNormalAtXi(mxi, master_normal.A());
  sele.DerivUnitNormalAtXi(sxi, deriv_slave_normal);
  mele.DerivUnitNormalAtXi(mxi, deriv_master_normal);

  double pen = ppn_;
  double pet = ppt_;

  const CORE::LINALG::Matrix<dim, 1> contact_normal(gpn, true);

  if (stype_ == INPAR::CONTACT::solution_nitsche)
  {
    double cauchy_nn_weighted_average = 0.;
    CORE::GEN::pairedvector<int, double> cauchy_nn_weighted_average_deriv(
        sele.NumNode() * 3 * 12 + sele.MoData().ParentDisp().size() +
        mele.MoData().ParentDisp().size());

    CORE::LINALG::SerialDenseVector normal_adjoint_test_slave(sele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_normal_adjoint_test_slave(
        sele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(sele.MoData().ParentDof().size(), true));

    CORE::LINALG::SerialDenseVector normal_adjoint_test_master(mele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_normal_adjoint_test_master(
        mele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(mele.MoData().ParentDof().size(), true));

    double ws = 0.;
    double wm = 0.;
    CONTACT::UTILS::NitscheWeightsAndScaling(sele, mele, nit_wgt_, dt_, ws, wm, pen, pet);

    // variables for friction (declaration only)
    CORE::LINALG::Matrix<dim, 1> t1, t2;
    std::vector<CORE::GEN::pairedvector<int, double>> dt1, dt2;
    CORE::LINALG::Matrix<dim, 1> relVel;
    std::vector<CORE::GEN::pairedvector<int, double>> relVel_deriv(
        dim, sele.NumNode() * dim + mele.NumNode() * dim + dsxi[0].size() + dmxi[0].size());
    double vt1(0.0), vt2(0.0);
    CORE::GEN::pairedvector<int, double> dvt1(0);
    CORE::GEN::pairedvector<int, double> dvt2(0);
    double cauchy_nt1_weighted_average = 0.;
    CORE::GEN::pairedvector<int, double> cauchy_nt1_weighted_average_deriv(
        sele.NumNode() * 3 * 12 + sele.MoData().ParentDisp().size() +
        mele.MoData().ParentDisp().size());
    CORE::LINALG::SerialDenseVector t1_adjoint_test_slave(sele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_t1_adjoint_test_slave(
        sele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(sele.MoData().ParentDof().size(), true));
    CORE::LINALG::SerialDenseVector t1_adjoint_test_master(mele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_t1_adjoint_test_master(
        mele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(mele.MoData().ParentDof().size(), true));
    double cauchy_nt2_weighted_average = 0.;
    CORE::GEN::pairedvector<int, double> cauchy_nt2_weighted_average_deriv(
        sele.NumNode() * 3 * 12 + sele.MoData().ParentDisp().size() +
        mele.MoData().ParentDisp().size());
    CORE::LINALG::SerialDenseVector t2_adjoint_test_slave(sele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_t2_adjoint_test_slave(
        sele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(sele.MoData().ParentDof().size(), true));
    CORE::LINALG::SerialDenseVector t2_adjoint_test_master(mele.MoData().ParentDof().size());
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector> deriv_t2_adjoint_test_master(
        mele.MoData().ParentDof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        CORE::LINALG::SerialDenseVector(mele.MoData().ParentDof().size(), true));
    double sigma_nt1_pen_vt1(0.0), sigma_nt2_pen_vt2(0.0);
    CORE::GEN::pairedvector<int, double> d_sigma_nt1_pen_vt1(
        dgapgp.capacity() + cauchy_nn_weighted_average_deriv.capacity() +
            cauchy_nt1_weighted_average_deriv.capacity() + dvt1.capacity(),
        0, 0);
    CORE::GEN::pairedvector<int, double> d_sigma_nt2_pen_vt2(
        dgapgp.capacity() + cauchy_nn_weighted_average_deriv.capacity() +
            cauchy_nt2_weighted_average_deriv.capacity() + dvt2.capacity(),
        0, 0);
    // variables for friction (end)

    SoEleCauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, contact_normal,
        deriv_contact_normal, ws, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv,
        normal_adjoint_test_slave, deriv_normal_adjoint_test_slave);
    SoEleCauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, contact_normal,
        deriv_contact_normal, -wm, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv,
        normal_adjoint_test_master, deriv_normal_adjoint_test_master);

    const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
    CORE::GEN::pairedvector<int, double> d_snn_av_pen_gap(
        cauchy_nn_weighted_average_deriv.size() + dgapgp.size());
    for (const auto& p : cauchy_nn_weighted_average_deriv) d_snn_av_pen_gap[p.first] += p.second;
    for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;

    // evaluation of tangential stuff
    if (frtype_)
    {
      CONTACT::UTILS::BuildTangentVectors<dim>(
          contact_normal.A(), deriv_contact_normal, t1.A(), dt1, t2.A(), dt2);
      CONTACT::UTILS::RelVelInvariant<dim>(sele, sxi, dsxi, sval, sderiv, mele, mxi, dmxi, mval,
          mderiv, gap, dgapgp, relVel, relVel_deriv);
      CONTACT::UTILS::VectorScalarProduct<dim>(t1, dt1, relVel, relVel_deriv, vt1, dvt1);
      CONTACT::UTILS::VectorScalarProduct<dim>(t2, dt2, relVel, relVel_deriv, vt2, dvt2);

      SoEleCauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, t1, dt1, ws,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1_adjoint_test_slave,
          deriv_t1_adjoint_test_slave);
      SoEleCauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, t1, dt1, -wm,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1_adjoint_test_master,
          deriv_t1_adjoint_test_master);

      SoEleCauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, t2, dt2, ws,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2_adjoint_test_slave,
          deriv_t2_adjoint_test_slave);
      SoEleCauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, t2, dt2, -wm,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2_adjoint_test_master,
          deriv_t2_adjoint_test_master);
    }  // evaluation of tangential stuff

    if (frtype_)
    {
      IntegrateTest<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1, dt1);
      IntegrateTest<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2, dt2);
      if (!two_half_pass_)
      {
        IntegrateTest<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1, dt1);
        IntegrateTest<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2, dt2);
      }

      IntegrateAdjointTest<dim>(-theta_ / pet, jac, jacintcellmap, wgt, cauchy_nt1_weighted_average,
          cauchy_nt1_weighted_average_deriv, sele, t1_adjoint_test_slave,
          deriv_t1_adjoint_test_slave);
      IntegrateAdjointTest<dim>(-theta_ / pet, jac, jacintcellmap, wgt, cauchy_nt2_weighted_average,
          cauchy_nt2_weighted_average_deriv, sele, t2_adjoint_test_slave,
          deriv_t2_adjoint_test_slave);
      if (!two_half_pass_)
      {
        IntegrateAdjointTest<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
            cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, mele,
            t1_adjoint_test_master, deriv_t1_adjoint_test_master);
        IntegrateAdjointTest<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
            cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, mele,
            t2_adjoint_test_master, deriv_t2_adjoint_test_master);
      }
    }

    if (snn_av_pen_gap >= 0.)
    {
      IntegrateTest<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
          deriv_contact_normal);
      if (!two_half_pass_)
      {
        IntegrateTest<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
            deriv_contact_normal);
      }

      IntegrateAdjointTest<dim>(-theta_ / pen, jac, jacintcellmap, wgt, cauchy_nn_weighted_average,
          cauchy_nn_weighted_average_deriv, sele, normal_adjoint_test_slave,
          deriv_normal_adjoint_test_slave);
      if (!two_half_pass_)
      {
        IntegrateAdjointTest<dim>(-theta_ / pen, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, mele,
            normal_adjoint_test_master, deriv_normal_adjoint_test_master);
      }
    }
    else
    {
      // test in normal contact direction
      IntegrateTest<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
          deriv_contact_normal);
      if (!two_half_pass_)
      {
        IntegrateTest<dim>(+1., mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
            deriv_contact_normal);
      }

      IntegrateTest<dim>(-theta_2_ * pen, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, gap,
          dgapgp, contact_normal, deriv_contact_normal);
      if (!two_half_pass_)
      {
        IntegrateTest<dim>(+theta_2_ * pen, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, gap,
            dgapgp, contact_normal, deriv_contact_normal);
      }

      IntegrateAdjointTest<dim>(theta_, jac, jacintcellmap, wgt, gap, dgapgp, sele,
          normal_adjoint_test_slave, deriv_normal_adjoint_test_slave);
      if (!two_half_pass_)
      {
        IntegrateAdjointTest<dim>(theta_, jac, jacintcellmap, wgt, gap, dgapgp, mele,
            normal_adjoint_test_master, deriv_normal_adjoint_test_master);
      }

      if (frtype_)
      {
        double fr = 0.0;
        switch (frtype_)
        {
          case INPAR::CONTACT::friction_coulomb:
            fr = frcoeff_ * (-1.) * (snn_av_pen_gap);
            break;
          case INPAR::CONTACT::friction_tresca:
            fr = frbound_;
            break;
          default:
            dserror("why are you here???");
            break;
        }

        double tan_tr = sqrt(
            (cauchy_nt1_weighted_average + pet * vt1) * (cauchy_nt1_weighted_average + pet * vt1) +
            (cauchy_nt2_weighted_average + pet * vt2) * (cauchy_nt2_weighted_average + pet * vt2));

        // stick
        if (tan_tr < fr)
        {
          sigma_nt1_pen_vt1 = cauchy_nt1_weighted_average + pet * vt1;
          for (const auto& p : dvt1) d_sigma_nt1_pen_vt1[p.first] += pet * p.second;
          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            d_sigma_nt1_pen_vt1[p.first] += p.second;

          sigma_nt2_pen_vt2 = cauchy_nt2_weighted_average + pet * vt2;
          for (const auto& p : dvt2) d_sigma_nt2_pen_vt2[p.first] += pet * p.second;
          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            d_sigma_nt2_pen_vt2[p.first] += p.second;
        }
        // slip
        else
        {
          CORE::GEN::pairedvector<int, double> tmp_d(
              dgapgp.size() + cauchy_nn_weighted_average_deriv.size() +
                  cauchy_nt1_weighted_average_deriv.size() + dvt1.size(),
              0, 0);
          if (frtype_ == INPAR::CONTACT::friction_coulomb)
            for (const auto& p : d_snn_av_pen_gap) tmp_d[p.first] += -frcoeff_ / tan_tr * p.second;

          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt1_weighted_average + pet * vt1) * p.second;
          for (const auto& p : dvt1)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt1_weighted_average + pet * vt1) * (+pet) * p.second;

          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt2_weighted_average + pet * vt2) * p.second;
          for (const auto& p : dvt2)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt2_weighted_average + pet * vt2) * (+pet) * p.second;

          sigma_nt1_pen_vt1 = fr / tan_tr * (cauchy_nt1_weighted_average + pet * vt1);
          for (const auto& p : tmp_d)
            d_sigma_nt1_pen_vt1[p.first] += p.second * (cauchy_nt1_weighted_average + pet * vt1);
          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            d_sigma_nt1_pen_vt1[p.first] += fr / tan_tr * p.second;
          for (const auto& p : dvt1) d_sigma_nt1_pen_vt1[p.first] += fr / tan_tr * pet * p.second;

          sigma_nt2_pen_vt2 = fr / tan_tr * (cauchy_nt2_weighted_average + pet * vt2);
          for (const auto& p : tmp_d)
            d_sigma_nt2_pen_vt2[p.first] += p.second * (cauchy_nt2_weighted_average + pet * vt2);
          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            d_sigma_nt2_pen_vt2[p.first] += fr / tan_tr * p.second;
          for (const auto& p : dvt2) d_sigma_nt2_pen_vt2[p.first] += fr / tan_tr * pet * p.second;
        }

        IntegrateTest<dim>(-theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
            sigma_nt1_pen_vt1, d_sigma_nt1_pen_vt1, t1, dt1);
        IntegrateTest<dim>(-theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
            sigma_nt2_pen_vt2, d_sigma_nt2_pen_vt2, t2, dt2);
        if (!two_half_pass_)
        {
          IntegrateTest<dim>(+theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
              sigma_nt1_pen_vt1, d_sigma_nt1_pen_vt1, t1, dt1);
          IntegrateTest<dim>(+theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
              sigma_nt2_pen_vt2, d_sigma_nt2_pen_vt2, t2, dt2);
        }

        IntegrateAdjointTest<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt1_pen_vt1,
            d_sigma_nt1_pen_vt1, sele, t1_adjoint_test_slave, deriv_t1_adjoint_test_slave);
        IntegrateAdjointTest<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt2_pen_vt2,
            d_sigma_nt2_pen_vt2, sele, t2_adjoint_test_slave, deriv_t2_adjoint_test_slave);
        if (!two_half_pass_)
        {
          IntegrateAdjointTest<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt1_pen_vt1,
              d_sigma_nt1_pen_vt1, mele, t1_adjoint_test_master, deriv_t1_adjoint_test_master);
          IntegrateAdjointTest<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt2_pen_vt2,
              d_sigma_nt2_pen_vt2, mele, t2_adjoint_test_master, deriv_t2_adjoint_test_master);
        }
      }
    }
  }
  else if ((stype_ == INPAR::CONTACT::solution_penalty) ||
           stype_ == INPAR::CONTACT::solution_multiscale)
  {
    if (gap < 0.)
    {
      IntegrateTest<dim>(-pen, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, gap, dgapgp,
          contact_normal, deriv_contact_normal);
      if (!two_half_pass_)
      {
        IntegrateTest<dim>(+pen, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, gap, dgapgp,
            contact_normal, deriv_contact_normal);
      }
    }
  }
  else
    dserror("unknown algorithm");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType parentdistype, int dim>
void inline CONTACT::UTILS::SoEleGP(MORTAR::MortarElement& sele, const double wgt,
    const double* gpcoord, CORE::LINALG::Matrix<dim, 1>& pxsi,
    CORE::LINALG::Matrix<dim, dim>& derivtrafo)
{
  CORE::DRT::UTILS::CollectedGaussPoints intpoints =
      CORE::DRT::UTILS::CollectedGaussPoints(1);  // reserve just for 1 entry ...
  intpoints.Append(gpcoord[0], gpcoord[1], 0.0, wgt);

  // get coordinates of gauss point w.r.t. local parent coordinate system
  CORE::LINALG::SerialDenseMatrix pqxg(1, dim);
  derivtrafo.Clear();

  CORE::DRT::UTILS::BoundaryGPToParentGP<dim>(pqxg, derivtrafo, intpoints,
      sele.ParentElement()->Shape(), sele.Shape(), sele.FaceParentNumber());

  // coordinates of the current integration point in parent coordinate system
  for (int idim = 0; idim < dim; idim++) pxsi(idim) = pqxg(0, idim);
}


template <int dim>
void CONTACT::UTILS::MapGPtoParent(MORTAR::MortarElement& moEle, double* boundary_gpcoord,
    const double wgt, CORE::LINALG::Matrix<dim, 1>& pxsi,
    CORE::LINALG::Matrix<dim, dim>& derivtravo_slave)
{
  DRT::Element::DiscretizationType distype = moEle.ParentElement()->Shape();
  switch (distype)
  {
    case DRT::Element::hex8:
      CONTACT::UTILS::SoEleGP<DRT::Element::hex8, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case DRT::Element::tet4:
      CONTACT::UTILS::SoEleGP<DRT::Element::tet4, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case DRT::Element::quad4:
      CONTACT::UTILS::SoEleGP<DRT::Element::quad4, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case DRT::Element::quad9:
      CONTACT::UTILS::SoEleGP<DRT::Element::quad9, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case DRT::Element::tri3:
      CONTACT::UTILS::SoEleGP<DRT::Element::tri3, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case DRT::Element::nurbs27:
      CONTACT::UTILS::SoEleGP<DRT::Element::nurbs27, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    default:
      dserror("Nitsche contact not implemented for used (bulk) elements");
  }
}


template <int dim>
void CONTACT::CoIntegratorNitsche::SoEleCauchy(MORTAR::MortarElement& moEle,
    double* boundary_gpcoord,
    std::vector<CORE::GEN::pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const CORE::LINALG::Matrix<dim, 1>& normal,
    std::vector<CORE::GEN::pairedvector<int, double>>& normal_deriv,
    const CORE::LINALG::Matrix<dim, 1>& direction,
    std::vector<CORE::GEN::pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, CORE::GEN::pairedvector<int, double>& deriv_sigma_nt,
    CORE::LINALG::SerialDenseVector& adjoint_test,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>& deriv_adjoint_test)
{
  CORE::LINALG::Matrix<dim, 1> pxsi(true);
  CORE::LINALG::Matrix<dim, dim> derivtravo_slave;
  CONTACT::UTILS::MapGPtoParent<dim>(moEle, boundary_gpcoord, gp_wgt, pxsi, derivtravo_slave);

  double sigma_nt = 0.0;
  CORE::LINALG::SerialDenseMatrix dsntdd, d2sntdd2, d2sntDdDn, d2sntDdDt, d2sntDdDpxi;
  CORE::LINALG::Matrix<dim, 1> dsntdn, dsntdt, dsntdpxi;
  dynamic_cast<DRT::ELEMENTS::So_base*>(moEle.ParentElement())
      ->GetCauchyNDirAndDerivativesAtXi(pxsi, moEle.MoData().ParentDisp(), normal, direction,
          sigma_nt, &dsntdd, &d2sntdd2, &d2sntDdDn, &d2sntDdDt, &d2sntDdDpxi, &dsntdn, &dsntdt,
          &dsntdpxi, nullptr, nullptr, nullptr, nullptr, nullptr);

  cauchy_nt += w * sigma_nt;

  for (int i = 0; i < moEle.ParentElement()->NumNode() * dim; ++i)
    deriv_sigma_nt[moEle.MoData().ParentDof().at(i)] += w * dsntdd(i, 0);

  for (int i = 0; i < dim - 1; ++i)
  {
    for (const auto& p : boundary_gpcoord_lin[i])
      for (int k = 0; k < dim; ++k)
        deriv_sigma_nt[p.first] += dsntdpxi(k) * derivtravo_slave(k, i) * p.second * w;
  }


  for (int d = 0; d < dim; ++d)
    for (const auto& p : normal_deriv[d]) deriv_sigma_nt[p.first] += dsntdn(d) * p.second * w;

  for (int d = 0; d < dim; ++d)
    for (const auto& p : direction_deriv[d]) deriv_sigma_nt[p.first] += dsntdt(d) * p.second * w;

  if (abs(theta_) > 1.e-12)
  {
    BuildAdjointTest<dim>(moEle, w, dsntdd, d2sntdd2, d2sntDdDn, d2sntDdDt, d2sntDdDpxi,
        boundary_gpcoord_lin, derivtravo_slave, normal_deriv, direction_deriv, adjoint_test,
        deriv_adjoint_test);
  }
}

template <int dim>
void CONTACT::CoIntegratorNitsche::IntegrateTest(const double fac, MORTAR::MortarElement& ele,
    const CORE::LINALG::SerialDenseVector& shape, const CORE::LINALG::SerialDenseMatrix& deriv,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dxi, const double jac,
    const CORE::GEN::pairedvector<int, double>& jacintcellmap, const double wgt,
    const double test_val, const CORE::GEN::pairedvector<int, double>& test_deriv,
    const CORE::LINALG::Matrix<dim, 1>& test_dir,
    const std::vector<CORE::GEN::pairedvector<int, double>>& test_dir_deriv)
{
  if (abs(fac) < 1.e-16) return;

  for (int d = 0; d < dim; ++d)
  {
    const double val = fac * jac * wgt * test_val * test_dir(d);

    for (int s = 0; s < ele.NumNode(); ++s)
    {
      *(ele.GetNitscheContainer().Rhs(CORE::DRT::UTILS::getParentNodeNumberFromFaceNodeNumber(
                                          ele.ParentElement()->Shape(), ele.FaceParentNumber(), s) *
                                          dim +
                                      d)) += val * shape(s);
    }

    std::unordered_map<int, double> val_deriv;

    for (const auto& p : jacintcellmap)
      val_deriv[p.first] += fac * p.second * wgt * test_val * test_dir(d);
    for (const auto& p : test_deriv) val_deriv[p.first] += fac * jac * wgt * test_dir(d) * p.second;
    for (const auto& p : test_dir_deriv[d])
      val_deriv[p.first] += fac * jac * wgt * test_val * p.second;

    for (const auto& p : val_deriv)
    {
      double* row = ele.GetNitscheContainer().K(p.first);
      for (int s = 0; s < ele.NumNode(); ++s)
      {
        row[CORE::DRT::UTILS::getParentNodeNumberFromFaceNodeNumber(
                ele.ParentElement()->Shape(), ele.FaceParentNumber(), s) *
                dim +
            d] += p.second * shape(s);
      }
    }

    for (int e = 0; e < dim - 1; ++e)
    {
      for (const auto& p : dxi[e])
      {
        double* row = ele.GetNitscheContainer().K(p.first);
        for (int s = 0; s < ele.NumNode(); ++s)
        {
          row[CORE::DRT::UTILS::getParentNodeNumberFromFaceNodeNumber(
                  ele.ParentElement()->Shape(), ele.FaceParentNumber(), s) *
                  dim +
              d] += val * deriv(s, e) * p.second;
        }
      }
    }
  }
}

template <int dim>
void CONTACT::CoIntegratorNitsche::BuildAdjointTest(MORTAR::MortarElement& moEle, const double fac,
    const CORE::LINALG::SerialDenseMatrix& dsntdd, const CORE::LINALG::SerialDenseMatrix& d2sntdd2,
    const CORE::LINALG::SerialDenseMatrix& d2sntDdDn,
    const CORE::LINALG::SerialDenseMatrix& d2sntDdDt,
    const CORE::LINALG::SerialDenseMatrix& d2sntDdDpxi,
    const std::vector<CORE::GEN::pairedvector<int, double>>& boundary_gpcoord_lin,
    CORE::LINALG::Matrix<dim, dim> derivtravo_slave,
    const std::vector<CORE::GEN::pairedvector<int, double>>& normal_deriv,
    const std::vector<CORE::GEN::pairedvector<int, double>>& direction_deriv,
    CORE::LINALG::SerialDenseVector& adjoint_test,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>& deriv_adjoint_test)
{
  for (int i = 0; i < moEle.ParentElement()->NumNode() * dim; ++i)
  {
    adjoint_test(i) = fac * dsntdd(i, 0);
    CORE::LINALG::SerialDenseVector& at = deriv_adjoint_test[moEle.MoData().ParentDof().at(i)];
    for (int j = 0; j < moEle.ParentElement()->NumNode() * dim; ++j) at(j) += fac * d2sntdd2(i, j);
  }

  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : normal_deriv[d])
    {
      CORE::LINALG::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.ParentElement()->NumNode() * dim; ++i)
        at(i) += fac * d2sntDdDn(i, d) * p.second;
    }
  }

  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : direction_deriv[d])
    {
      CORE::LINALG::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.ParentElement()->NumNode() * dim; ++i)
        at(i) += fac * d2sntDdDt(i, d) * p.second;
    }
  }

  CORE::LINALG::SerialDenseMatrix tmp(moEle.ParentElement()->NumNode() * dim, dim, false);
  CORE::LINALG::SerialDenseMatrix deriv_trafo(Teuchos::View, derivtravo_slave.A(),
      derivtravo_slave.numRows(), derivtravo_slave.numRows(), derivtravo_slave.numCols());
  if (tmp.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1., d2sntDdDpxi, deriv_trafo, 0.))
    dserror("multiply failed");
  for (int d = 0; d < dim - 1; ++d)
  {
    for (const auto& p : boundary_gpcoord_lin[d])
    {
      CORE::LINALG::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.ParentElement()->NumNode() * dim; ++i)
        at(i) += fac * tmp(i, d) * p.second;
    }
  }
}


template <int dim>
void CONTACT::CoIntegratorNitsche::IntegrateAdjointTest(const double fac, const double jac,
    const CORE::GEN::pairedvector<int, double>& jacintcellmap, const double wgt, const double test,
    const CORE::GEN::pairedvector<int, double>& deriv_test, MORTAR::MortarElement& moEle,
    CORE::LINALG::SerialDenseVector& adjoint_test,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>& deriv_adjoint_test)
{
  if (abs(fac) < 1.e-16) return;

  CORE::LINALG::SerialDenseVector Tmp(
      Teuchos::View, moEle.GetNitscheContainer().Rhs(), moEle.MoData().ParentDof().size());
  CORE::LINALG::Update(fac * jac * wgt * test, adjoint_test, 1.0, Tmp);

  for (const auto& p : deriv_adjoint_test)
  {
    CORE::LINALG::SerialDenseVector Tmp(
        Teuchos::View, moEle.GetNitscheContainer().K(p.first), moEle.MoData().ParentDof().size());
    CORE::LINALG::Update(fac * jac * wgt * test, p.second, 1.0, Tmp);
  }

  for (const auto& p : jacintcellmap)
  {
    CORE::LINALG::SerialDenseVector Tmp(
        Teuchos::View, moEle.GetNitscheContainer().K(p.first), moEle.MoData().ParentDof().size());
    CORE::LINALG::Update(fac * p.second * wgt * test, adjoint_test, 1.0, Tmp);
  }

  for (const auto& p : deriv_test)
  {
    CORE::LINALG::SerialDenseVector Tmp(
        Teuchos::View, moEle.GetNitscheContainer().K(p.first), moEle.MoData().ParentDof().size());
    CORE::LINALG::Update(fac * jac * wgt * p.second, adjoint_test, 1.0, Tmp);
  }
}

void CONTACT::UTILS::NitscheWeightsAndScaling(MORTAR::MortarElement& sele,
    MORTAR::MortarElement& mele, const INPAR::CONTACT::NitscheWeighting nit_wgt, const double dt,
    double& ws, double& wm, double& pen, double& pet)
{
  const double he_slave = dynamic_cast<CONTACT::CoElement&>(sele).TraceHE();
  const double he_master = dynamic_cast<CONTACT::CoElement&>(mele).TraceHE();

  switch (nit_wgt)
  {
    case INPAR::CONTACT::NitWgt_slave:
    {
      ws = 1.;
      wm = 0.;
      pen /= he_slave;
      pet /= he_slave;
    }
    break;
    case INPAR::CONTACT::NitWgt_master:
    {
      wm = 1.;
      ws = 0.;
      pen /= he_master;
      pet /= he_master;
    }
    break;
    case INPAR::CONTACT::NitWgt_harmonic:
      ws = 1. / he_master;
      wm = 1. / he_slave;
      ws /= (ws + wm);
      wm = 1. - ws;
      pen = ws * pen / he_slave + wm * pen / he_master;
      pet = ws * pet / he_slave + wm * pet / he_master;

      break;
    default:
      dserror("unknown Nitsche weighting");
      break;
  }
}

template <int dim>
void CONTACT::UTILS::RelVel(MORTAR::MortarElement& ele,
    const CORE::LINALG::SerialDenseVector& shape, const CORE::LINALG::SerialDenseMatrix& deriv,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dxi, const double fac,
    CORE::LINALG::Matrix<dim, 1>& relVel,
    std::vector<CORE::GEN::pairedvector<int, double>>& relVel_deriv)
{
  for (int n = 0; n < ele.NumNode(); ++n)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) += fac * shape(n) * (ele.GetNodalCoords(d, n) - ele.GetNodalCoordsOld(d, n));
      relVel_deriv[d][dynamic_cast<MORTAR::MortarNode*>(ele.Nodes()[n])->Dofs()[d]] +=
          fac * shape(n);

      for (int sd = 0; sd < dim - 1; ++sd)
      {
        for (const auto& p : dxi[sd])
        {
          relVel_deriv[d][p.first] += fac *
                                      (ele.GetNodalCoords(d, n) - ele.GetNodalCoordsOld(d, n)) *
                                      deriv(n, sd) * p.second;
        }
      }
    }
  }
}


template <int dim>
void CONTACT::UTILS::RelVelInvariant(MORTAR::MortarElement& sele, const double* sxi,
    const std::vector<CORE::GEN::pairedvector<int, double>>& derivsxi,
    const CORE::LINALG::SerialDenseVector& sval, const CORE::LINALG::SerialDenseMatrix& sderiv,
    MORTAR::MortarElement& mele, const double* mxi,
    const std::vector<CORE::GEN::pairedvector<int, double>>& derivmxi,
    const CORE::LINALG::SerialDenseVector& mval, const CORE::LINALG::SerialDenseMatrix& mderiv,
    const double& gap, const CORE::GEN::pairedvector<int, double>& deriv_gap,
    CORE::LINALG::Matrix<dim, 1>& relVel,
    std::vector<CORE::GEN::pairedvector<int, double>>& relVel_deriv, const double fac)
{
  CORE::LINALG::Matrix<3, 1> n_old;
  CORE::LINALG::Matrix<3, 2> d_n_old_dxi;
  dynamic_cast<CONTACT::CoElement&>(sele).OldUnitNormalAtXi(sxi, n_old, d_n_old_dxi);
  for (int i = 0; i < sele.NumNode(); ++i)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) += sele.GetNodalCoordsOld(d, i) * sval(i) * fac;

      for (int e = 0; e < dim - 1; ++e)
        for (const auto& p : derivsxi[e])
          relVel_deriv[d][p.first] += sele.GetNodalCoordsOld(d, i) * sderiv(i, e) * p.second * fac;
    }
  }

  for (int i = 0; i < mele.NumNode(); ++i)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) -= mele.GetNodalCoordsOld(d, i) * mval(i) * fac;

      for (int e = 0; e < dim - 1; ++e)
        for (const auto& p : derivmxi[e])
          relVel_deriv[d][p.first] -= mele.GetNodalCoordsOld(d, i) * mderiv(i, e) * p.second * fac;
    }
  }
  for (int d = 0; d < dim; ++d)
  {
    relVel(d) += n_old(d) * gap * fac;

    for (int e = 0; e < dim - 1; ++e)
      for (const auto& p : derivsxi[e])
        relVel_deriv[d][p.first] += gap * d_n_old_dxi(d, e) * p.second * fac;

    for (const auto& p : deriv_gap) relVel_deriv[d][p.first] += n_old(d) * p.second * fac;
  }
}

template <int dim>
void CONTACT::UTILS::VectorScalarProduct(const CORE::LINALG::Matrix<dim, 1>& v1,
    const std::vector<CORE::GEN::pairedvector<int, double>>& v1d,
    const CORE::LINALG::Matrix<dim, 1>& v2,
    const std::vector<CORE::GEN::pairedvector<int, double>>& v2d, double& val,
    CORE::GEN::pairedvector<int, double>& val_deriv)
{
  val = v1.Dot(v2);
  val_deriv.clear();
  val_deriv.resize(v1d[0].size() + v2d[0].size());
  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : v1d[d]) val_deriv[p.first] += v2(d) * p.second;
    for (const auto& p : v2d[d]) val_deriv[p.first] += v1(d) * p.second;
  }
}

void CONTACT::UTILS::BuildTangentVectors3D(const double* np,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dn, double* t1p,
    std::vector<CORE::GEN::pairedvector<int, double>>& dt1, double* t2p,
    std::vector<CORE::GEN::pairedvector<int, double>>& dt2)
{
  const CORE::LINALG::Matrix<3, 1> n(np, false);
  CORE::LINALG::Matrix<3, 1> t1(t1p, true);
  CORE::LINALG::Matrix<3, 1> t2(t2p, true);

  bool z = true;
  CORE::LINALG::Matrix<3, 1> tmp;
  tmp(2) = 1.;
  if (abs(tmp.Dot(n)) > 1. - 1.e-4)
  {
    tmp(0) = 1.;
    tmp(2) = 0.;
    z = false;
  }

  t1.CrossProduct(tmp, n);
  dt1.resize(3, std::max(dn[0].size(), std::max(dn[1].size(), dn[2].size())));
  dt2.resize(3, std::max(dn[0].size(), std::max(dn[1].size(), dn[2].size())));

  const double lt1 = t1.Norm2();
  t1.Scale(1. / lt1);
  CORE::LINALG::Matrix<3, 3> p;
  for (int i = 0; i < 3; ++i) p(i, i) = 1.;
  p.MultiplyNT(-1., t1, t1, 1.);
  p.Scale(1. / lt1);
  if (z)
  {
    for (const auto& i : dn[1])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] -= p(d, 0) * i.second;

    for (const auto& i : dn[0])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] += p(d, 1) * i.second;
  }
  else
  {
    for (const auto& i : dn[2])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] -= p(d, 1) * i.second;

    for (const auto& i : dn[1])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] += p(d, 2) * i.second;
  }

  t2.CrossProduct(n, t1);
  if (abs(t2.Norm2() - 1.) > 1.e-10) dserror("this should already form an orthonormal basis");

  for (const auto& i : dn[0])
  {
    dt2[1][i.first] -= t1(2) * (i.second);
    dt2[2][i.first] += t1(1) * (i.second);
  }
  for (const auto& i : dn[1])
  {
    dt2[0][i.first] += t1(2) * (i.second);
    dt2[2][i.first] -= t1(0) * (i.second);
  }
  for (const auto& i : dn[2])
  {
    dt2[0][i.first] -= t1(1) * (i.second);
    dt2[1][i.first] += t1(0) * (i.second);
  }
  for (const auto& i : dt1[0])
  {
    dt2[1][i.first] += n(2) * (i.second);
    dt2[2][i.first] -= n(1) * (i.second);
  }
  for (const auto& i : dt1[1])
  {
    dt2[0][i.first] -= n(2) * (i.second);
    dt2[2][i.first] += n(0) * (i.second);
  }
  for (const auto& i : dt1[2])
  {
    dt2[0][i.first] += n(1) * (i.second);
    dt2[1][i.first] -= n(0) * (i.second);
  }
}

template <int dim>
void CONTACT::UTILS::BuildTangentVectors(const double* np,
    const std::vector<CORE::GEN::pairedvector<int, double>>& dn, double* t1p,
    std::vector<CORE::GEN::pairedvector<int, double>>& dt1, double* t2p,
    std::vector<CORE::GEN::pairedvector<int, double>>& dt2)
{
  if (dim == 3)
    BuildTangentVectors3D(np, dn, t1p, dt1, t2p, dt2);
  else
    dserror("not implemented");
}

template void CONTACT::UTILS::BuildTangentVectors<2>(const double*,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, double*,
    std::vector<CORE::GEN::pairedvector<int, double>>&, double*,
    std::vector<CORE::GEN::pairedvector<int, double>>&);

template void CONTACT::UTILS::BuildTangentVectors<3>(const double*,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, double*,
    std::vector<CORE::GEN::pairedvector<int, double>>&, double*,
    std::vector<CORE::GEN::pairedvector<int, double>>&);



template void CONTACT::CoIntegratorNitsche::IntegrateTest<2>(const double, MORTAR::MortarElement&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>& i, const double,
    const CORE::GEN::pairedvector<int, double>&, const double, const double,
    const CORE::GEN::pairedvector<int, double>&, const CORE::LINALG::Matrix<2, 1>& test_dir,
    const std::vector<CORE::GEN::pairedvector<int, double>>& test_dir_deriv);
template void CONTACT::CoIntegratorNitsche::IntegrateTest<3>(const double, MORTAR::MortarElement&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>& i, const double,
    const CORE::GEN::pairedvector<int, double>&, const double, const double,
    const CORE::GEN::pairedvector<int, double>&, const CORE::LINALG::Matrix<3, 1>& test_dir,
    const std::vector<CORE::GEN::pairedvector<int, double>>& test_dir_deriv);

template void CONTACT::CoIntegratorNitsche::IntegrateAdjointTest<2>(const double, const double,
    const CORE::GEN::pairedvector<int, double>&, const double, const double,
    const CORE::GEN::pairedvector<int, double>&, MORTAR::MortarElement&,
    CORE::LINALG::SerialDenseVector&,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>&);

template void CONTACT::CoIntegratorNitsche::IntegrateAdjointTest<3>(const double, const double,
    const CORE::GEN::pairedvector<int, double>&, const double, const double,
    const CORE::GEN::pairedvector<int, double>&, MORTAR::MortarElement&,
    CORE::LINALG::SerialDenseVector&,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>&);

template void CONTACT::CoIntegratorNitsche::BuildAdjointTest<2>(MORTAR::MortarElement&,
    const double, const CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::SerialDenseMatrix&,
    const CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::SerialDenseMatrix&,
    const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, CORE::LINALG::Matrix<2, 2>,
    const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, CORE::LINALG::SerialDenseVector&,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>&);

template void CONTACT::CoIntegratorNitsche::BuildAdjointTest<3>(MORTAR::MortarElement&,
    const double, const CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::SerialDenseMatrix&,
    const CORE::LINALG::SerialDenseMatrix&, const CORE::LINALG::SerialDenseMatrix&,
    const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, CORE::LINALG::Matrix<3, 3>,
    const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, CORE::LINALG::SerialDenseVector&,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseVector>&);


template void CONTACT::UTILS::RelVel<2>(MORTAR::MortarElement&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, const double,
    CORE::LINALG::Matrix<2, 1>&, std::vector<CORE::GEN::pairedvector<int, double>>&);

template void CONTACT::UTILS::RelVel<3>(MORTAR::MortarElement&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, const double,
    CORE::LINALG::Matrix<3, 1>&, std::vector<CORE::GEN::pairedvector<int, double>>&);

template void CONTACT::UTILS::VectorScalarProduct<2>(const CORE::LINALG::Matrix<2, 1>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, const CORE::LINALG::Matrix<2, 1>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, double&,
    CORE::GEN::pairedvector<int, double>&);
template void CONTACT::UTILS::VectorScalarProduct<3>(const CORE::LINALG::Matrix<3, 1>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, const CORE::LINALG::Matrix<3, 1>&,
    const std::vector<CORE::GEN::pairedvector<int, double>>&, double&,
    CORE::GEN::pairedvector<int, double>&);

template void CONTACT::UTILS::RelVelInvariant<2>(MORTAR::MortarElement&, const double*,
    const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    MORTAR::MortarElement&, const double*, const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&, const double&,
    const CORE::GEN::pairedvector<int, double>&, CORE::LINALG::Matrix<2, 1>&,
    std::vector<CORE::GEN::pairedvector<int, double>>&, const double);

template void CONTACT::UTILS::RelVelInvariant<3>(MORTAR::MortarElement&, const double*,
    const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&,
    MORTAR::MortarElement&, const double*, const std::vector<CORE::GEN::pairedvector<int, double>>&,
    const CORE::LINALG::SerialDenseVector&, const CORE::LINALG::SerialDenseMatrix&, const double&,
    const CORE::GEN::pairedvector<int, double>&, CORE::LINALG::Matrix<3, 1>&,
    std::vector<CORE::GEN::pairedvector<int, double>>&, const double);

template void CONTACT::UTILS::MapGPtoParent<2>(MORTAR::MortarElement&, double*, const double,
    CORE::LINALG::Matrix<2, 1>&, CORE::LINALG::Matrix<2, 2>&);

template void CONTACT::UTILS::MapGPtoParent<3>(MORTAR::MortarElement&, double*, const double,
    CORE::LINALG::Matrix<3, 1>&, CORE::LINALG::Matrix<3, 3>&);
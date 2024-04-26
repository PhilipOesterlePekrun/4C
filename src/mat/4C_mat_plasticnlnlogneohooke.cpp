/*----------------------------------------------------------------------*/
/*! \file
\brief Contains the functions to establish local material law /
       stress-strain law for isotropic material following finite strain
       von-Mises plasticity with linear isotropic hardening
       and logarithmic hyperelastic material (i.e. linear relation
       between Kirchhoff-stress and logarithmic strain; aka Hencky
       material model).
       The principal stress based implementation follows
       Bonet and Wood: "Nonlinear continuum mechanics for finite element analysis."
       Cambridge University Press, Cambridge, 2008

       geometrically nonlinear, finite strains, rate-independent

       example input line:
       MAT 1 MAT_Struct_PlasticNlnLogNeoHooke YOUNG 206.9 NUE 0.29 DENS 0.0
         YIELD 0.45 ISOHARD 0.12924 SATHARDENING 0.715 HARDEXPO 16.93 VISC 1.0 RATE_DEPENDENCY 0.1

\level 2

*/
/*----------------------------------------------------------------------*
 | headers                                                              |
 *----------------------------------------------------------------------*/
#include "4C_mat_plasticnlnlogneohooke.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::pair<double, double> ResiduumAndJacobianFromFunction(
      MAT::PAR::PlasticNlnLogNeoHooke* matparameter, double Dgamma, double dt, double accplstrain,
      double abs_dev_KH_trial, const CORE::UTILS::FunctionOfAnything& hardeningfunction)
  {
    const double ym = matparameter->youngs_;        // Young's modulus
    const double nu = matparameter->poissonratio_;  // Poisson's ratio
    const double G = ym / (2.0 * (1.0 + nu));       // shear modulus, mu=G
    // plastic material data
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    //! vector for input of accumulated strain to function
    std::vector<std::pair<std::string, double>> dp;
    dp.emplace_back("epsp", accplstrain + Dgamma);

    const double y_d = hardeningfunction.Evaluate(dp, {}, 0);
    double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);

    const std::vector<double> dy_d_dgvector = hardeningfunction.EvaluateDerivative(dp, {}, 0);
    const double dy_d_dgamma = dy_d_dgvector[0] * pow(visc * Dgamma / dt + 1., eps) +
                               y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    double residuum = sqrt(3.0 / 2.0) * abs_dev_KH_trial - 3. * G * Dgamma - y_d_visc;
    double tangent = -3. * G - dy_d_dgamma;
    return {residuum, tangent};
  }

  std::pair<double, double> ResiduumAndJacobian(MAT::PAR::PlasticNlnLogNeoHooke* matparameter,
      double Dgamma, double dt, double accplstrain_last, double abs_dev_KH_trial)
  {
    const double ym = matparameter->youngs_;        // Young's modulus
    const double nu = matparameter->poissonratio_;  // Poisson's ratio
    const double G = ym / (2.0 * (1.0 + nu));       // shear modulus, mu=G
    // plastic material data
    const double yield = matparameter->yield_;          // initial yield stress
    const double isohard = matparameter->isohard_;      // linear isotropic hardening
    const double infyield = matparameter->infyield_;    // saturation yield stress
    const double hardexp = matparameter->hardexp_;      // nonlinear hardening exponent
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    double accplstrain_curr = accplstrain_last + Dgamma;

    const double y_d = (yield + isohard * accplstrain_curr +
                        (infyield - yield) * (1. - exp(-hardexp * accplstrain_curr)));
    const double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);
    const double dy_d_dgamma =
        (isohard + (infyield - yield) * hardexp * exp(-hardexp * accplstrain_curr)) *
            pow(visc * Dgamma / dt + 1., eps) +
        y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    double residuum = sqrt(3.0 / 2.0) * abs_dev_KH_trial - 3. * G * Dgamma - y_d_visc;
    double tangent = -3. * G - dy_d_dgamma;
    return {residuum, tangent};
  }

  std::pair<double, double> SolveNewtonWithHardeningFunc(
      MAT::PAR::PlasticNlnLogNeoHooke* matparameter, double abs_dev_KH_trial,
      double accplstrain_last, double dt, const CORE::UTILS::FunctionOfAnything& hardening_function,
      bool error_tol)
  {
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    // local Newton iteration for nonlinear isotropic hardening
    const int maxiter = matparameter->max_iterations_;
    const double tol = matparameter->tolerance_nr_;

    auto residuumAndJacobianFromFunction = [&](double Dgamma)
    {
      return ResiduumAndJacobianFromFunction(
          matparameter, Dgamma, dt, accplstrain_last, abs_dev_KH_trial, hardening_function);
    };

    const double Dgamma =
        CORE::UTILS::SolveLocalNewton(residuumAndJacobianFromFunction, 0.0, tol, maxiter);

    //! vector for input of accumulated strain to function
    std::vector<std::pair<std::string, double>> dp;
    dp.emplace_back("epsp", accplstrain_last + Dgamma);
    const double y_d = hardening_function.Evaluate(dp, {}, 0);

    std::vector<double> dy_d_dgvector = hardening_function.EvaluateDerivative(dp, {}, 0);
    double dy_d_dgamma = dy_d_dgvector[0] * pow(visc * Dgamma / dt + 1., eps) +
                         y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    return {Dgamma, dy_d_dgamma};
  }

  std::pair<double, double> SolveNewtonWithParameters(MAT::PAR::PlasticNlnLogNeoHooke* matparameter,
      double abs_dev_KH_trial, double accplstrain_last, double dt, bool error_tol)
  {
    // plastic material data
    const double yield = matparameter->yield_;          // initial yield stress
    const double isohard = matparameter->isohard_;      // linear isotropic hardening
    const double infyield = matparameter->infyield_;    // saturation yield stress
    const double hardexp = matparameter->hardexp_;      // nonlinear hardening exponent
    const double visc = matparameter->visc_;            // viscosity
    const double eps = matparameter->rate_dependency_;  // rate dependency

    // local Newton iteration for nonlinear isotropic hardening
    const int maxiter = matparameter->max_iterations_;
    const double tol = matparameter->tolerance_nr_;

    auto residuumAndJacobian = [&](double Dgamma)
    { return ResiduumAndJacobian(matparameter, Dgamma, dt, accplstrain_last, abs_dev_KH_trial); };

    const double Dgamma = CORE::UTILS::SolveLocalNewton(residuumAndJacobian, 0.0, tol, maxiter);

    const double accplstrain_curr = accplstrain_last + Dgamma;

    const double y_d = (yield + isohard * accplstrain_curr +
                        (infyield - yield) * (1. - exp(-hardexp * accplstrain_curr)));
    const double dy_d_dgamma =
        (isohard + (infyield - yield) * hardexp * exp(-hardexp * accplstrain_curr)) *
            pow(visc * Dgamma / dt + 1., eps) +
        y_d * eps * pow(visc * Dgamma / dt + 1., eps - 1) * visc / dt;

    return {Dgamma, dy_d_dgamma};
  }

}  // namespace



/*----------------------------------------------------------------------*
 | constructor (public)                                                 |
 *----------------------------------------------------------------------*/
MAT::PAR::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke(Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      youngs_(*matdata->Get<double>("YOUNG")),
      poissonratio_(*matdata->Get<double>("NUE")),
      density_(*matdata->Get<double>("DENS")),
      yield_(*matdata->Get<double>("YIELD")),
      isohard_(*matdata->Get<double>("ISOHARD")),
      infyield_(*matdata->Get<double>("SATHARDENING")),
      hardexp_(*matdata->Get<double>("HARDEXPO")),
      visc_(*matdata->Get<double>("VISC")),
      rate_dependency_(*matdata->Get<double>("RATE_DEPENDENCY")),
      functionID_hardening_(*matdata->Get<int>("HARDENING_FUNC")),
      max_iterations_(10),
      tolerance_nr_(1.e-12)
{
  if (yield_ == 0 && functionID_hardening_ == 0)
    FOUR_C_THROW(
        "You have to provide either a parameter for "
        "HARDENING_FUNC or YIELD in MAT_Struct_PlasticNlnLogNeoHooke");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from ReadMaterials()                  |
 *----------------------------------------------------------------------*/
Teuchos::RCP<MAT::Material> MAT::PAR::PlasticNlnLogNeoHooke::CreateMaterial()
{
  return Teuchos::rcp(new MAT::PlasticNlnLogNeoHooke(this));
}


MAT::PlasticNlnLogNeoHookeType MAT::PlasticNlnLogNeoHookeType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from ReadMaterials()                  |
 *----------------------------------------------------------------------*/
CORE::COMM::ParObject* MAT::PlasticNlnLogNeoHookeType::Create(const std::vector<char>& data)
{
  MAT::PlasticNlnLogNeoHooke* plasticneo = new MAT::PlasticNlnLogNeoHooke();
  plasticneo->Unpack(data);
  return plasticneo;
}


/*----------------------------------------------------------------------*
 | constructor (public)                                                 |
 *----------------------------------------------------------------------*/
MAT::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke() : params_(nullptr) {}


/*----------------------------------------------------------------------*
 | copy-constructor (public)                                            |
 *----------------------------------------------------------------------*/
MAT::PlasticNlnLogNeoHooke::PlasticNlnLogNeoHooke(MAT::PAR::PlasticNlnLogNeoHooke* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*
 | pack (public)                                                        |
 *----------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);

  // pack history data
  int histsize;
  // if material is not initialized, i.e. start simulation, nothing to pack
  if (!Initialized())
  {
    histsize = 0;
  }
  else
  {
    // if material is initialized (restart): size equates number of gausspoints
    histsize = accplstrainlast_.size();
  }
  AddtoPack(data, histsize);  // Length of history vector(s)
  for (int var = 0; var < histsize; ++var)
  {
    // insert history vectors to AddtoPack
    AddtoPack(data, accplstrainlast_.at(var));
    AddtoPack(data, invplrcglast_.at(var));
  }

  return;
}  // Pack()


/*----------------------------------------------------------------------*
 | unpack (public)                                                      |
 *----------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::Unpack(const std::vector<char>& data)
{
  isinit_ = true;
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (GLOBAL::Problem::Instance()->Materials() != Teuchos::null)
    if (GLOBAL::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = GLOBAL::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          GLOBAL::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::PlasticNlnLogNeoHooke*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  // history data
  int histsize;
  ExtractfromPack(position, data, histsize);

  // if system is not yet initialized, the history vectors have to be intialized
  if (histsize == 0) isinit_ = false;

  for (int var = 0; var < histsize; ++var)
  {
    double tmp1 = 0.0;
    // scalar-valued vector of last converged state are unpacked
    ExtractfromPack(position, data, tmp1);
    accplstrainlast_.push_back(tmp1);

    CORE::LINALG::Matrix<3, 3> tmp(true);
    // vectors of last converged state are unpacked
    ExtractfromPack(position, data, tmp);
    invplrcglast_.push_back(tmp);

    // current vectors have to be initialized
    accplstraincurr_.push_back(tmp1);
    invplrcgcurr_.push_back(tmp);
  }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", data.size(), position);

  return;

}  // Unpack()


/*---------------------------------------------------------------------*
 | initialise / allocate internal variables (public)                   |
 *---------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::Setup(int numgp, INPUT::LineDefinition* linedef)
{
  // Extract the function for hardening only once because this is expensive.
  const int functionID_hardening =
      params_->functionID_hardening_;  // function number for isotropic hardening
  if (functionID_hardening != 0)
  {
    hardening_function_ =
        &GLOBAL::Problem::Instance()->FunctionById<CORE::UTILS::FunctionOfAnything>(
            functionID_hardening - 1);
  }

  invplrcglast_.resize(numgp);
  invplrcgcurr_.resize(numgp);

  accplstrainlast_.resize(numgp);
  accplstraincurr_.resize(numgp);

  CORE::LINALG::Matrix<3, 3> emptymat(true);
  for (int i = 0; i < 3; i++) emptymat(i, i) = 1.0;

  for (int i = 0; i < numgp; i++)
  {
    invplrcglast_.at(i) = emptymat;
    invplrcgcurr_.at(i) = emptymat;

    accplstrainlast_.at(i) = 0.0;
    accplstraincurr_.at(i) = 0.0;
  }

  isinit_ = true;
}  // Setup()


/*----------------------------------------------------------------------*
 | update internal variables                                            |
 *----------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::Update()
{
  // make current values at time step t_n+1 to values of last step t_n
  invplrcglast_ = invplrcgcurr_;
  accplstrainlast_ = accplstraincurr_;

  // empty vectors of current data
  invplrcgcurr_.clear();

  // get the size of the vector
  // (use the last vector, because it includes latest results, current is empty)
  const int histsize = invplrcglast_.size();
  invplrcgcurr_.resize(histsize);
  accplstraincurr_.resize(histsize);

  CORE::LINALG::Matrix<3, 3> emptymat(true);
  for (int i = 0; i < 3; i++) emptymat(i, i) = 1.0;

  for (int i = 0; i < histsize; i++)
  {
    invplrcgcurr_.at(i) = emptymat;
    accplstraincurr_.at(i) = 0.0;
  }
  return;
}  // Update()


/*----------------------------------------------------------------------*
 | calculate stress and constitutive tensor                             |
 *----------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
    const CORE::LINALG::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    CORE::LINALG::Matrix<6, 1>* stress, CORE::LINALG::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  // elastic material data
  // get material parameters
  const double ym = params_->youngs_;                  // Young's modulus
  const double nu = params_->poissonratio_;            // Poisson's ratio
  const double G = ym / (2.0 * (1.0 + nu));            // shear modulus, mu=G
  const double kappa = ym / (3.0 * (1.0 - 2.0 * nu));  // bulk modulus

  // plastic material data
  const double yield = params_->yield_;          // initial yield stress
  const double isohard = params_->isohard_;      // linear isotropic hardening
  const double infyield = params_->infyield_;    // saturation yield stress
  const double hardexp = params_->hardexp_;      // nonlinear hardening exponent
  const double visc = params_->visc_;            // viscosity
  const double eps = params_->rate_dependency_;  // rate dependency

  const double detF = defgrd->Determinant();

  const double dt = params.get<double>("delta time");
  // check, if errors are tolerated or should throw a FOUR_C_THROW
  bool error_tol = false;
  if (params.isParameter("tolerate_errors")) error_tol = params.get<bool>("tolerate_errors");

  // plastic increment
  double Dgamma = 0.0;
  double dy_d_dgamma = 0.0;

  CORE::LINALG::Matrix<3, 3> invdefgrd(*defgrd);
  invdefgrd.Invert();

  // matrices for temporary stuff
  CORE::LINALG::Matrix<3, 3> tmp1;
  CORE::LINALG::Matrix<3, 3> tmp2;

  // 3x3 2nd-order identity matrix
  CORE::LINALG::Matrix<3, 3> id2(true);
  // 3x3 2nd-order deviatoric identity matrix in principal directions
  CORE::LINALG::Matrix<3, 3> Idev;
  for (int i = 0; i < 3; i++)
  {
    id2(i, i) = 1.0;
    for (int j = 0; j < 3; j++)
    {
      if (i == j)
        Idev(i, j) = 2.0 / 3.0;
      else
        Idev(i, j) = -1.0 / 3.0;
    }
  }

  // linear elasticity tensor in principal directions
  CORE::LINALG::Matrix<3, 3> D_ep_principal(Idev);
  D_ep_principal.Scale(2.0 * G);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) D_ep_principal(i, j) += kappa;

  // ------------------------------------------------------- trial strain
  // elastic left Cauchy-Green deformation tensor (LCG) at trial state
  // Be_trial = b_{e,n+1}^{trial} = F_{n+1} C_{p,n}^{-1} F_{n+1}
  CORE::LINALG::Matrix<3, 3> Be_trial;
  tmp1.Multiply(*defgrd, invplrcglast_.at(gp));
  Be_trial.MultiplyNT(tmp1, *defgrd);

  // elastic LCG at final state
  CORE::LINALG::Matrix<3, 3> Be;

  // ***************************************************
  // Here we start the principal stress based algorithm
  // ***************************************************

  // convert to epetra format and solve eigenvalue problem
  // this matrix contains spatial eigen directions (the second
  // index corresponds to the eigenvalue)
  CORE::LINALG::SerialDenseMatrix n(3, 3);
  CORE::LINALG::SerialDenseVector lambda_trial_square(3);

  // convert Input Matrix in Epetra format
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) n(i, j) = Be_trial(i, j);

  // calculate eigenvectors and eigenvalues
  CORE::LINALG::SymmetricEigenProblem(n, lambda_trial_square);
  // eigenvectors are stored in n, i.e. original matrix inserted in method is
  // destroyed.
  // eigenvalues correspond to the square of the stretches

  // spatial principal direction n_alpha (in CORE::LINALG format)
  std::vector<CORE::LINALG::Matrix<3, 1>> spatial_principal_directions;
  spatial_principal_directions.resize(3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (spatial_principal_directions.at(i))(j) = n(j, i);

  // material principal directions N_alpha
  // note, that for convenience in the later programming this is NOT
  // based on the correct pull-back N_alpha = lambda_alpha * F^-1 * n_alpha
  // Instead we just do N_alpha = F^{-1} * n_alpha
  std::vector<CORE::LINALG::Matrix<3, 1>> material_principal_directions;
  material_principal_directions.resize(3);
  for (int i = 0; i < 3; i++)
    material_principal_directions.at(i).Multiply(invdefgrd, spatial_principal_directions.at(i));

  // deviatoric Kirchhoff stress at trial state
  // tau^{trial} = 2 * G * log(sqrt(lambda^2)) - 2/3 * G * log(detF)
  CORE::LINALG::Matrix<3, 1> dev_KH_trial;
  for (int i = 0; i < 3; i++)
    dev_KH_trial(i) = G * std::log(lambda_trial_square(i)) - 2.0 / 3.0 * G * std::log(detF);

  // deviatoric Kirchhoff stress at final state
  // For now we store the trial states. In case of plastic
  // material reaction, we will update it later
  CORE::LINALG::Matrix<3, 1> dev_KH;

  // pressure (equal at trial state and final state) for the use with tau
  // tau = tau'_aa + p = tau'_aa + kappa * log(detF)
  // later used for S = F^{-1} . tau . F^{-T}, no scaling with detF required
  double pressure = kappa * std::log(detF);

  // ----------------------------------------------------- yield function
  // calculate von Mises equivalent stress at trial state
  double abs_dev_KH_trial = 0.0;
  for (int i = 0; i < 3; i++) abs_dev_KH_trial += dev_KH_trial(i) * dev_KH_trial(i);
  abs_dev_KH_trial = std::sqrt(abs_dev_KH_trial);

  //! vector for input of differential pressure to function
  std::vector<std::pair<std::string, double>> dp;
  dp.emplace_back("epsp", accplstrainlast_.at(gp));
  const double y_d = hardening_function_
                         ? hardening_function_->Evaluate(dp, {}, 0)
                         : yield + isohard * accplstrainlast_.at(gp) +
                               (infyield - yield) * (1. - exp(-hardexp * accplstrainlast_.at(gp)));


  const double y_d_visc = y_d * pow(visc * Dgamma / dt + 1., eps);

  const double f_trial = std::sqrt(3.0 / 2.0) * abs_dev_KH_trial - y_d_visc;

  // switch elastic (f <= 0) and plastic (f > 0) state
  if (f_trial <= 0.0)  // ----------------------------------- elastic step
  {
    // trial state variables are final variables
    dev_KH.Update(dev_KH_trial);
    Be.Update(Be_trial);

    // plastic increment is zero
    Dgamma = 0.0;
  }

  else  // -------------------------------------------------- plastic step
  {
    std::pair<double, double> gamma_and_derivative =
        (hardening_function_) ? SolveNewtonWithHardeningFunc(params_, abs_dev_KH_trial,
                                    accplstrainlast_.at(gp), dt, *hardening_function_, error_tol)
                              : SolveNewtonWithParameters(params_, abs_dev_KH_trial,
                                    accplstrainlast_.at(gp), dt, error_tol);

    Dgamma = gamma_and_derivative.first;
    if (Dgamma < 0.0)
    {
      params.set<bool>("eval_error", true);
      return;
    }
    // flow vector (7.77) in de Souza Neta
    CORE::LINALG::Matrix<3, 1> flow_vector(dev_KH_trial);
    flow_vector.Scale(1.0 / (sqrt(2.0 / 3.0) * abs_dev_KH_trial));

    // stress return mapping (7.89) in de Souza Neto
    double fac_dev_KH = 1.0 - 2.0 * G * Dgamma / (std::sqrt(2.0 / 3.0) * abs_dev_KH_trial);
    dev_KH.Update(fac_dev_KH, dev_KH_trial, 0.0);

    // strain return mapping
    // b_{e,n+1} = sum_i^3 ( lambda_{n+1}^2 . n \otimes n )
    // lambda_{n+1} = lambda_{n+1}^{trial} / exp(Dgamma . flow_vector)
    Be.Clear();
    for (int i = 0; i < 3; i++)
    {
      tmp1.MultiplyNT(spatial_principal_directions.at(i), spatial_principal_directions.at(i));
      Be.Update((lambda_trial_square(i) / exp(2.0 * flow_vector(i) * Dgamma)), tmp1, 1.0);
    }

    // update tangent modulus
    dy_d_dgamma = gamma_and_derivative.second;
    double fac_D_ep_1 = -4.0 * G * G * Dgamma / (sqrt(2.0 / 3.0) * abs_dev_KH_trial);
    D_ep_principal.Update(fac_D_ep_1, Idev, 1.0);
    tmp1.MultiplyNT(flow_vector, flow_vector);
    double fac_D_ep_2 =
        4.0 * G * G * (sqrt(2.0 / 3.0) * Dgamma / abs_dev_KH_trial - 1.0 / (3.0 * G + dy_d_dgamma));
    D_ep_principal.Update(fac_D_ep_2, tmp1, 1.0);
  }

  // -------------------------------------------------- output PK2 stress
  // tau_ij = tau_ii * n_i \otimes n_i
  // or tau_ij = tau'_ii + kappa ln(J)
  // S_ij   = F^-1 tau_ij F^-T
  //        = tau_ij (F^-1 n_i) \otimes (F^-1 n_i)
  CORE::LINALG::Matrix<3, 3> PK2(true);
  for (int i = 0; i < 3; i++)
  {
    tmp1.MultiplyNT(material_principal_directions.at(i), material_principal_directions.at(i));
    PK2.Update((dev_KH(i) + pressure), tmp1, 1.0);
  }

  // output stress in Voigt notation
  (*stress)(0) = PK2(0, 0);
  (*stress)(1) = PK2(1, 1);
  (*stress)(2) = PK2(2, 2);
  (*stress)(3) = 0.5 * (PK2(0, 1) + PK2(1, 0));
  (*stress)(4) = 0.5 * (PK2(1, 2) + PK2(2, 1));
  (*stress)(5) = 0.5 * (PK2(2, 0) + PK2(0, 2));

  // ---------------------------------------------------- tangent modulus
  // express coefficents of tangent in Kirchhoff stresses
  cmat->Clear();
  for (int a = 0; a < 3; a++)
  {
    // - sum_1^3 (2 * tau N_aaaa)
    tmp1.MultiplyNT(
        material_principal_directions.at(a), material_principal_directions.at(a));  // N_{aa}
    ElastSymTensorMultiply(*cmat, -2.0 * (dev_KH(a) + pressure), tmp1, tmp1, 1.0);

    for (int b = 0; b < 3; b++)
    {
      // c_ab N_aabb
      // result of return mapping of deviatoric component c_ab
      tmp1.MultiplyNT(
          material_principal_directions.at(a), material_principal_directions.at(a));  // N_{aa}
      tmp2.MultiplyNT(
          material_principal_directions.at(b), material_principal_directions.at(b));  // N_{bb}
      ElastSymTensorMultiply(*cmat, D_ep_principal(a, b), tmp1, tmp2, 1.0);

      if (a != b)
      {
        const double fac =
            (lambda_trial_square(a) != lambda_trial_square(b))
                ?  // (tau_aa * lambda_b^2 - tau_bb * lambda_a^2) / (lambda_a^2 - lambda_b^2)
                ((dev_KH(a) + pressure) * lambda_trial_square(b) -
                    (dev_KH(b) + pressure) * lambda_trial_square(a)) /
                    (lambda_trial_square(a) - lambda_trial_square(b))
                :  // 1/2 [(d^2 Psi)/(d ln lambda_b * d ln lambda_b) -
                   //                (d^2 Psi)/(d ln lambda_a * d ln lambda_b)]
                   // - tau_bb, cf. (6.91)
                0.5 * (D_ep_principal(b, b) - D_ep_principal(a, b)) - (dev_KH(b) + pressure);
        tmp1.MultiplyNT(
            material_principal_directions.at(a), material_principal_directions.at(b));  // N_{ab}
        tmp2.MultiplyNT(
            material_principal_directions.at(b), material_principal_directions.at(a));  // N_{ba}
        ElastSymTensorMultiply(*cmat, fac, tmp1, tmp1, 1.0);                            // N_{abab}
        ElastSymTensorMultiply(*cmat, fac, tmp1, tmp2, 1.0);                            // N_{abba}

      }  // end if (a!=b)
    }    // end loop b
  }      // end loop a

  // --------------------------------------------- update plastic history
  // plastic inverse of right Cauchy-Green deformation tensor (RCG)
  tmp1.Multiply(invdefgrd, Be);
  invplrcgcurr_.at(gp).MultiplyNT(tmp1, invdefgrd);

  // accumulated plastic strain
  accplstraincurr_.at(gp) = accplstrainlast_.at(gp) + Dgamma;

  // Green-Lagrange plastic strains can be easily calculated, in contrast
  // Euler-Almansi requires special treatment, which is not yet considered in the
  // element formulation

  return;

}  // Evaluate()


/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::VisNames(std::map<std::string, int>& names)
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar

}  // VisNames()


/*---------------------------------------------------------------------*
 | return names of visualization data for direct VTK output            |
 *---------------------------------------------------------------------*/
void MAT::PlasticNlnLogNeoHooke::RegisterOutputDataNames(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["accumulated_plastic_strain"] = 1;
  names_and_size["active_plasticity"] = 1;
  names_and_size["plastic_strain"] = 6;
}


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool MAT::PlasticNlnLogNeoHooke::VisData(
    const std::string& name, std::vector<double>& data, int numgp, int eleID)
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double temp = 0.0;
    for (int gp = 0; gp < numgp; ++gp) temp += AccumulatedStrain(gp);
    data[0] = temp / numgp;
  }
  return false;

}  // VisData()


bool MAT::PlasticNlnLogNeoHooke::EvaluateOutputData(
    const std::string& name, CORE::LINALG::SerialDenseMatrix& data) const
{
  if (name == "accumulated_plastic_strain")
  {
    for (std::size_t gp = 0; gp < accplstraincurr_.size(); ++gp)
    {
      data(gp, 0) = accplstraincurr_.at(int(gp));
    }
    return true;
  }
  if (name == "plastic_strain")
  {
    for (std::size_t gp = 0; gp < invplrcgcurr_.size(); ++gp)
    {
      const double* values = invplrcgcurr_.at(gp).A();
      for (std::size_t i = 0; i < 6; ++i)
      {
        data(gp, i) = values[i];
      }
    }
    return true;
  }
  if (name == "active_plasticity")
  {
    for (std::size_t gp = 0; gp < invplrcgcurr_.size(); ++gp)
    {
      data(gp, 0) = (accplstraincurr_.at(int(gp)) > accplstrainlast_.at(int(gp))) ? 1 : 0;
    }
    return true;
  }
  return false;
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
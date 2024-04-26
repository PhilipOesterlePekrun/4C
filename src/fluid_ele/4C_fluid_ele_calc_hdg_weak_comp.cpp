/*----------------------------------------------------------------------------*/
/*! \file
\brief Routines for calculation of HDG weakly compressible fluid element

\level 2

*/
/*----------------------------------------------------------------------------*/

#include "4C_fluid_ele_calc_hdg_weak_comp.hpp"

#include "4C_discretization_fem_general_extract_values.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_calc.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_functions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_fluid_weakly_compressible.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN



template <CORE::FE::CellType distype>
DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::FluidEleCalcHDGWeakComp() : usescompletepoly_(true)
{
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::InitializeShapes(
    const DRT::ELEMENTS::Fluid* ele)
{
  // Check if this is an HDG weakly compressible element
  if (const DRT::ELEMENTS::FluidHDGWeakComp* hdgwkele =
          dynamic_cast<const DRT::ELEMENTS::FluidHDGWeakComp*>(ele))
  {
    // use complete polynomial space
    usescompletepoly_ = hdgwkele->UsesCompletePolynomialSpace();

    // initialize shapes
    if (shapes_ == Teuchos::null)
      shapes_ = Teuchos::rcp(new CORE::FE::ShapeValues<distype>(
          hdgwkele->Degree(), usescompletepoly_, 2 * ele->Degree()));
    else if (shapes_->degree_ != unsigned(ele->Degree()) ||
             shapes_->usescompletepoly_ != usescompletepoly_)
      shapes_ = Teuchos::rcp(new CORE::FE::ShapeValues<distype>(
          hdgwkele->Degree(), usescompletepoly_, 2 * ele->Degree()));

    // initialize shapes on faces
    if (shapesface_ == Teuchos::null)
    {
      CORE::FE::ShapeValuesFaceParams svfparams(
          ele->Degree(), usescompletepoly_, 2 * ele->Degree());
      shapesface_ = Teuchos::rcp(new CORE::FE::ShapeValuesFace<distype>(svfparams));
    }

    // initialize local solver
    if (local_solver_ == Teuchos::null)
      local_solver_ = Teuchos::rcp(new LocalSolver(ele, *shapes_, *shapesface_, usescompletepoly_));
  }
  else
    FOUR_C_THROW("Only works for HDG weakly compressible fluid elements");
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::Evaluate(DRT::ELEMENTS::Fluid* ele,
    DRT::Discretization& discretization, const std::vector<int>& lm, Teuchos::ParameterList& params,
    Teuchos::RCP<MAT::Material>& mat, CORE::LINALG::SerialDenseMatrix& elemat1_epetra,
    CORE::LINALG::SerialDenseMatrix& elemat2_epetra,
    CORE::LINALG::SerialDenseVector& elevec1_epetra,
    CORE::LINALG::SerialDenseVector& elevec2_epetra,
    CORE::LINALG::SerialDenseVector& elevec3_epetra, const CORE::FE::GaussIntegration&,
    bool offdiag)
{
  return this->Evaluate(ele, discretization, lm, params, mat, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra, offdiag);
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::Evaluate(DRT::ELEMENTS::Fluid* ele,
    DRT::Discretization& discretization, const std::vector<int>& lm, Teuchos::ParameterList& params,
    Teuchos::RCP<MAT::Material>& mat, CORE::LINALG::SerialDenseMatrix& elemat1,
    CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseVector& elevec1,
    CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&, bool offdiag)
{
  // read global vectors
  ReadGlobalVectors(*ele, discretization, lm);

  // initialize shapes
  InitializeShapes(ele);
  shapes_->Evaluate(*ele, ale_dis_);

  // initialize all
  local_solver_->InitializeAll();

  // compute interior residual and matrices
  local_solver_->ComputeInteriorResidual(mat, interior_val_, interior_acc_, ale_vel_);
  local_solver_->ComputeInteriorMatrices(mat);

  // compute face residual and face matrices
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    shapesface_->EvaluateFace(*ele, f, ale_dis_);
    local_solver_->ComputeFaceResidual(f, mat, interior_val_, trace_val_, ale_vel_);
    local_solver_->ComputeFaceMatrices(f, mat);
  }

  // group residuals
  local_solver_->ComputeLocalResidual();
  local_solver_->ComputeGlobalResidual(*ele);

  // group matrices
  local_solver_->ComputeLocalLocalMatrix();
  local_solver_->ComputeLocalGlobalMatrix(*ele);
  local_solver_->ComputeGlobalLocalMatrix(*ele);
  local_solver_->ComputeGlobalGlobalMatrix(*ele);

  // condense local part
  local_solver_->InvertLocalLocalMatrix();
  local_solver_->CondenseLocalResidual(elevec1);
  local_solver_->CondenseLocalMatrix(elemat1);

  // divide rhs by alpha_f
  elevec1.scale(1.0 / local_solver_->fldparatimint_->AlphaF());

  return 0;
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::ReadGlobalVectors(
    const DRT::Element& ele, DRT::Discretization& discretization, const std::vector<int>& lm)
{
  // initialize the vectors
  trace_val_.resize(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  interior_val_.resize((msd_ + 1 + nsd_) * shapes_->ndofs_);
  interior_acc_.resize((msd_ + 1 + nsd_) * shapes_->ndofs_);

  // read the trace values
  Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(0, "velaf");
  CORE::FE::ExtractMyValues(*matrix_state, trace_val_, lm);

  // get local dofs
  std::vector<int> localDofs = discretization.Dof(1, &ele);

  // read the interior values
  matrix_state = discretization.GetState(1, "intvelaf");
  CORE::FE::ExtractMyValues(*matrix_state, interior_val_, localDofs);

  // read the interior time derivatives
  matrix_state = discretization.GetState(1, "intaccam");
  CORE::FE::ExtractMyValues(*matrix_state, interior_acc_, localDofs);

  // read ale vectors
  ReadAleVectors(ele, discretization);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::ReadAleVectors(
    const DRT::Element& ele, DRT::Discretization& discretization)
{
  // initialize ale vectors
  ale_dis_.resize(nsd_ * nen_);
  ale_vel_.resize(nsd_ * nen_);

  // check presence of ale state
  if (discretization.HasState(2, "dispnp"))
  {
    const DRT::ELEMENTS::FluidHDGWeakComp* hdgwkele =
        dynamic_cast<const DRT::ELEMENTS::FluidHDGWeakComp*>(&ele);
    if (hdgwkele->IsAle())
    {
      // get ale dofs
      const int* nodeids = ele.NodeIds();
      std::vector<int> aleDofs;
      for (unsigned int n = 0; n < nen_; ++n)
      {
        std::vector<int> nodeDofs = discretization.Dof(2, discretization.gNode(nodeids[n]));
        for (unsigned int d = 0; d < nsd_; ++d) aleDofs.push_back(nodeDofs[d]);
      }

      // initialize matrix state
      Teuchos::RCP<const Epetra_Vector> matrix_state;

      // read the ale displacement
      matrix_state = discretization.GetState(2, "dispnp");
      CORE::FE::ExtractMyValues(*matrix_state, ale_dis_, aleDofs);

      // read the ale velocity
      matrix_state = discretization.GetState(2, "gridv");
      CORE::FE::ExtractMyValues(*matrix_state, ale_vel_, aleDofs);
    }
  }
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::EvaluateService(DRT::ELEMENTS::Fluid* ele,
    Teuchos::ParameterList& params, Teuchos::RCP<MAT::Material>& mat,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseMatrix& elemat2,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseVector& elevec2,
    CORE::LINALG::SerialDenseVector& elevec3)
{
  // get the action required
  const FLD::Action act = CORE::UTILS::GetAsEnum<FLD::Action>(params, "action");

  switch (act)
  {
    case FLD::project_fluid_field:
    {
      return ProjectField(ele, params, mat, discretization, lm, elevec1, elevec2);
      break;
    }
    case FLD::update_local_solution:
    {
      return UpdateLocalSolution(ele, params, mat, discretization, lm, elevec1);
      break;
    }
    case FLD::interpolate_hdg_to_node:
    {
      return InterpolateSolutionToNodes(ele, discretization, elevec1);
      break;
    }
    case FLD::calc_fluid_error:
    {
      return ComputeError(ele, params, mat, discretization, lm, elevec1);
      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action for FluidHDG");
      break;
  }

  return 0;
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::UpdateLocalSolution(DRT::ELEMENTS::Fluid* ele,
    Teuchos::ParameterList& params, Teuchos::RCP<MAT::Material>& mat,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& interiorinc)
{
  // read global vectors
  ReadGlobalVectors(*ele, discretization, lm);

  // initialize shapes
  InitializeShapes(ele);
  shapes_->Evaluate(*ele, ale_dis_);

  // initialize all
  local_solver_->InitializeAll();

  // compute interior residual and matrices
  local_solver_->ComputeInteriorResidual(mat, interior_val_, interior_acc_, ale_vel_);
  local_solver_->ComputeInteriorMatrices(mat);

  // compute face residual and face matrices
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    shapesface_->EvaluateFace(*ele, f, ale_dis_);
    local_solver_->ComputeFaceResidual(f, mat, interior_val_, trace_val_, ale_vel_);
    local_solver_->ComputeFaceMatrices(f, mat);
  }

  // group residuals
  local_solver_->ComputeLocalResidual();

  // group matrices
  local_solver_->ComputeLocalLocalMatrix();
  local_solver_->ComputeLocalGlobalMatrix(*ele);

  // invert local-local matrix
  local_solver_->InvertLocalLocalMatrix();

  // extract local trace increments
  std::vector<double> localtraceinc_vec;
  localtraceinc_vec.resize(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(0, "globaltraceinc");
  CORE::FE::ExtractMyValues(*matrix_state, localtraceinc_vec, lm);

  // convert local trace increments to CORE::LINALG::SerialDenseVector
  CORE::LINALG::SerialDenseVector localtraceinc(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  for (unsigned int i = 0; i < nfaces_ * (1 + nsd_) * shapesface_->nfdofs_; ++i)
    localtraceinc(i) = localtraceinc_vec[i];

  // compute local solver vector
  CORE::LINALG::SerialDenseVector LocalSolverVec((msd_ + 1 + nsd_) * shapes_->ndofs_);
  CORE::LINALG::multiply(LocalSolverVec, local_solver_->KlocallocalInv, local_solver_->Rlocal);

  // compute local solver matrix
  CORE::LINALG::SerialDenseMatrix LocalSolverMat(
      (msd_ + 1 + nsd_) * shapes_->ndofs_, nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  CORE::LINALG::multiply(
      0.0, LocalSolverMat, -1.0, local_solver_->KlocallocalInv, local_solver_->Klocalglobal);

  // compute local increments
  interiorinc.size((msd_ + 1 + nsd_) * shapes_->ndofs_);
  interiorinc = LocalSolverVec;
  CORE::LINALG::multiply(1.0, interiorinc, 1.0, LocalSolverMat, localtraceinc);

  return 0;
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::ComputeError(DRT::ELEMENTS::Fluid* ele,
    Teuchos::ParameterList& params, Teuchos::RCP<MAT::Material>& mat,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec)
{
  // read ale vectors
  ReadAleVectors(*ele, discretization);

  // initialize shapes
  InitializeShapes(ele);
  shapes_->Evaluate(*ele, ale_dis_);

  // get time
  const double time = local_solver_->fldparatimint_->Time();

  // get interior values
  Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(1, "intvelnp");
  std::vector<int> localDofs = discretization.Dof(1, ele);
  std::vector<double> vecValues(localDofs.size());
  for (unsigned int i = 0; i < localDofs.size(); ++i)
  {
    const int lid = matrix_state->Map().LID(localDofs[i]);
    vecValues[i] = (*matrix_state)[lid];
  }

  // initialize exact solution
  CORE::LINALG::Matrix<msd_, 1> L_ex(true);
  double r_ex = 0.0;
  CORE::LINALG::Matrix<nsd_, 1> w_ex(true);

  // initialize spatial coordinates
  CORE::LINALG::Matrix<nsd_, 1> xyz(true);

  // get function number
  const int calcerrfunctno =
      CORE::UTILS::GetAsEnum<INPAR::FLUID::CalcError>(params, "error function number");

  // initialize errors
  double err_L = 0.0;
  double err_r = 0.0;
  double err_w = 0.0;
  double norm_L = 0.0;
  double norm_r = 0.0;
  double norm_w = 0.0;

  // initialize numerical values
  CORE::LINALG::SerialDenseVector Leg;
  CORE::LINALG::SerialDenseVector reg;
  CORE::LINALG::SerialDenseVector weg;

  // ease notation
  CORE::LINALG::SerialDenseMatrix N = shapes_->shfunct;
  CORE::LINALG::SerialDenseVector fac = shapes_->jfac;
  unsigned int nqpoints = shapes_->nqpoints_;
  unsigned int ndofs = shapes_->ndofs_;

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // clear vectors
    Leg.size(msd_);
    reg.size(1);
    weg.size(nsd_);

    // interpolate values on gauss points
    for (unsigned int i = 0; i < ndofs; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) Leg(m) += N(i, q) * vecValues[m * ndofs + i];

      reg(0) += N(i, q) * vecValues[msd_ * ndofs + i];

      for (unsigned int d = 0; d < nsd_; ++d)
        weg(d) += N(i, q) * vecValues[(msd_ + 1 + d) * ndofs + i];
    }

    for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_->xyzreal(d, q);

    // evaluate analytical solution
    EvaluateAll(calcerrfunctno, xyz, time, L_ex, r_ex, w_ex);

    // evaluate errors
    for (unsigned int m = 0; m < msd_; ++m)
      err_L += (L_ex(m) - Leg(m)) * (L_ex(m) - Leg(m)) * fac(q);

    err_r += (r_ex - reg(0)) * (r_ex - reg(0)) * fac(q);

    for (unsigned int d = 0; d < nsd_; ++d)
      err_w += (w_ex(d) - weg(d)) * (w_ex(d) - weg(d)) * fac(q);

    // evaluate norms
    for (unsigned int m = 0; m < msd_; ++m) norm_L += L_ex(m) * L_ex(m) * fac(q);

    norm_r += r_ex * r_ex * fac(q);

    for (unsigned int d = 0; d < nsd_; ++d) norm_w += w_ex(d) * w_ex(d) * fac(q);
  }

  // fill vector
  elevec[0] += err_L;
  elevec[1] += err_r;
  elevec[2] += err_w;
  elevec[3] += norm_L;
  elevec[4] += norm_r;
  elevec[5] += norm_w;

  return 0;
};



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::ProjectField(DRT::ELEMENTS::Fluid* ele,
    Teuchos::ParameterList& params, Teuchos::RCP<MAT::Material>& mat,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseVector& elevec2)
{
  // read ale vectors
  ReadAleVectors(*ele, discretization);

  // Create the necessary objects to the solution of the problem as the solver
  // and the shape functions for both the interior, shapes_, and the trace, shapesface_.
  InitializeShapes(ele);

  // Evaluate the element at the gauss points
  shapes_->Evaluate(*ele, ale_dis_);

  // reshape elevec2 as matrix
  FOUR_C_ASSERT(elevec2.numRows() == 0 ||
                    elevec2.numRows() == static_cast<int>((msd_ + 1 + nsd_) * shapes_->ndofs_),
      "Wrong size in project vector 2");

  // get initial function and current time
  const int* initfield = params.getPtr<int>("initfield");
  const int* startfunc = params.getPtr<int>("startfuncno");
  double* time = params.getPtr<double>("time");

  if (elevec2.numRows() > 0)
  {
    // Create the local matrix from starting at the addres where elevec2 is with the right shape
    CORE::LINALG::SerialDenseMatrix localMat(
        Teuchos::View, elevec2.values(), shapes_->ndofs_, shapes_->ndofs_, msd_ + 1 + nsd_);
    // Initialize matrix to zeros
    localMat.putScalar(0.0);

    // create mass matrix for interior by looping over quadrature points
    for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
    {
      // jfac is a vector containing the jacobian times the weight of the quadrature points
      const double fac = shapes_->jfac(q);
      // xyz is a vector containing the coordiantes of the quadrature points in real coordinates
      CORE::LINALG::Matrix<nsd_, 1> xyz(false);
      // Filling xyz with the values take from the element xyzreal matrix
      for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_->xyzreal(d, q);
      // Declaring vectors for interior variables
      CORE::LINALG::Matrix<msd_, 1> L(true);
      double r = 0.0;
      CORE::LINALG::Matrix<nsd_, 1> w(true);

      FOUR_C_ASSERT(initfield != nullptr && startfunc != nullptr,
          "initfield or startfuncno not set for initial value");

      // This function returns th value of the interior variables from the
      // given initial field that can be a know field or a user-defined one
      EvaluateDensityMomentum(*startfunc, xyz, 0.0, r, w);

      // now fill the components in the one-sided mass matrix and the right hand side
      for (unsigned int i = 0; i < shapes_->ndofs_; ++i)
      {
        local_solver_->massPart(i, q) = shapes_->shfunct(i, q);
        local_solver_->massPartW(i, q) = shapes_->shfunct(i, q) * fac;

        // RHS for the mixed variable
        for (unsigned int m = 0; m < msd_; ++m)
          localMat(i, m) += shapes_->shfunct(i, q) * L(m) * fac;

        // RHS for the density
        localMat(i, msd_) += shapes_->shfunct(i, q) * r * fac;

        // RHS for the momenutm
        for (unsigned int d = 0; d < nsd_; ++d)
          localMat(i, msd_ + 1 + d) += shapes_->shfunct(i, q) * w(d) * fac;
      }
    }
    // The integration is made by computing the matrix product
    CORE::LINALG::multiplyNT(
        local_solver_->massMat, local_solver_->massPart, local_solver_->massPartW);
    using ordinalType = CORE::LINALG::SerialDenseMatrix::ordinalType;
    using scalarType = CORE::LINALG::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(local_solver_->massMat));
    inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
    inverseMass.solve();
  }

  // Here we have the projection of the field on the trace
  CORE::LINALG::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
  // trVec is the vector of the trace values
  CORE::LINALG::SerialDenseMatrix trVec(shapesface_->nfdofs_, 1 + nsd_);
  FOUR_C_ASSERT(
      elevec1.numRows() == static_cast<int>((1 + nsd_) * shapesface_->nfdofs_) ||
          elevec1.numRows() == static_cast<int>(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_),
      "Wrong size in project vector 1");

  const unsigned int* faceConsider = params.getPtr<unsigned int>("faceconsider");
  Teuchos::Array<int>* functno = params.getPtr<Teuchos::Array<int>>("funct");
  Teuchos::Array<int>* onoff = params.getPtr<Teuchos::Array<int>>("onoff");

  // Project the field for all the faces of the element
  for (unsigned int face = 0; face < nfaces_; ++face)
  {
    // check whether we are in the project phase for all faces or for boundary values
    if (initfield == nullptr)
    {
      // We get here only if it is not an initial value but it is a time
      // dependant boundary value. If we are here we only want the function to run
      // for boundary faces specified in the faceConsider variable
      FOUR_C_ASSERT(faceConsider != nullptr, "Unsupported operation");
      if (*faceConsider != face) continue;
    }

    // the same function as before but for the trace elements
    shapesface_->EvaluateFace(*ele, face, ale_dis_);

    // Initializing the matrices
    mass.putScalar(0.0);
    trVec.putScalar(0.0);

    // For each quadrature point we evaluate the trace values and the shape functions
    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      // shapesface_->jfac contains the jacobian evaluated in the quadrature points
      const double fac = shapesface_->jfac(q);
      // xyz is the vector containing the coordinates of the quadrature points
      CORE::LINALG::Matrix<nsd_, 1> xyz(false);

      // Taking the real coordinates of quadrature points of the current face
      for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapesface_->xyzreal(d, q);

      // Creating the vector of traces variables
      double r;
      CORE::LINALG::Matrix<nsd_, 1> w(false);

      // Create dummy variables
      double dummy_r;
      CORE::LINALG::Matrix<nsd_, 1> dummy_w;

      // Deciding if we are initializing a field or if it is a time dependant
      // boundary condition
      if (initfield != nullptr)  // Initial function
        EvaluateDensityMomentum(*startfunc, xyz, 0.0, r, w);
      else
      {
        // This is used to project a function only on the boundary during the
        // temporal evolution of the simulation
        FOUR_C_ASSERT(functno != nullptr && time != nullptr && onoff != nullptr,
            "No array with functions given");

        // Deciding if to use the function or not for the density
        if ((*onoff)[0] == 1)
        {
          const int funct_num = (*functno)[0];
          if (funct_num > 0) EvaluateDensityMomentum(funct_num, xyz, *time, r, dummy_w);
        }

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          // Deciding if to use the function or not for the current component of the momentum
          if ((*onoff)[1 + d] == 0) continue;

          // If we are using the function, evaluate it
          const int funct_num = (*functno)[1 + d];
          if (funct_num > 0) EvaluateDensityMomentum(funct_num, xyz, *time, dummy_r, w);
        }
      }
      // now fill the components in the mass matrix and the right hand side

      // This is a more usual way to compute the mass matrix (double for loop)
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // mass matrix
        for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
          mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;

        // RHS
        trVec(i, 0) += shapesface_->shfunct(i, q) * r * fac;
        for (unsigned int d = 0; d < nsd_; ++d)
          trVec(i, 1 + d) += shapesface_->shfunct(i, q) * w(d) * fac;
      }
    }

    // Solving step, nothing fancy
    using ordinalType = CORE::LINALG::SerialDenseMatrix::ordinalType;
    using scalarType = CORE::LINALG::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(mass));
    // In this cas trVec is a proper vector and not a matrix used as multiple
    // RHS vectors
    inverseMass.setVectors(Teuchos::rcpFromRef(trVec), Teuchos::rcpFromRef(trVec));
    inverseMass.solve();

    // In this case we fill elevec1 with the values of trVec because we have not
    // defined trVec as a matrix beginning where elevec1 begins
    if (initfield != nullptr)  // This is for initial functions
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        elevec1(face * shapesface_->nfdofs_ * (1 + nsd_) + i) = trVec(i, 0);
        for (unsigned int d = 0; d < nsd_; ++d)
          elevec1(face * shapesface_->nfdofs_ * (1 + nsd_) + (1 + d) * shapesface_->nfdofs_ + i) =
              trVec(i, 1 + d);
      }
    else  // This is only for boundary faces during time evolution
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        elevec1(i) = trVec(i, 0);
        for (unsigned int d = 0; d < nsd_; ++d)
          elevec1((1 + d) * shapesface_->nfdofs_ + i) = trVec(i, 1 + d);
      }
  }  // for over the faces

  return 0;
}



template <CORE::FE::CellType distype>
int DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::InterpolateSolutionToNodes(
    DRT::ELEMENTS::Fluid* ele, DRT::Discretization& discretization,
    CORE::LINALG::SerialDenseVector& elevec1)
{
  // read ale vectors
  ReadAleVectors(*ele, discretization);

  InitializeShapes(ele);
  // Check if the vector has the correct size
  FOUR_C_ASSERT(elevec1.numRows() == (int)nen_ * (msd_ + 1 + nsd_ + 1 + nsd_),
      "Vector does not have correct size");

  // Getting the connectivity matrix
  // Contains the (local) coordinates of the nodes belonging to the element
  CORE::LINALG::SerialDenseMatrix locations =
      CORE::FE::getEleNodeNumbering_nodes_paramspace(distype);

  // This vector will contain the values of the shape functions computed in a
  // certain coordinate. In fact the lenght of the vector is given by the number
  // of shape functions, that is the same of the number of degrees of freedom of
  // an element.
  CORE::LINALG::SerialDenseVector values(shapes_->ndofs_);

  // get local solution values
  // The vector "matrix_state" contains the interior velocity values following
  // the local id numbers
  Teuchos::RCP<const Epetra_Vector> matrix_state = discretization.GetState(1, "intvelnp");
  // Vector of the ids of the DOF for the element
  std::vector<int> localDofs = discretization.Dof(1, ele);
  // SOLution VALUES
  std::vector<double> solvalues(localDofs.size());

  // Filling every entry of the solvalue vector obtaining the values from the
  //"matrix_state" vector.
  for (unsigned int i = 0; i < solvalues.size(); ++i)
  {
    // Finding the local id of the current "localDofs"
    const int lid = matrix_state->Map().LID(localDofs[i]);
    // Saving the value of the "localDofs[i]" in the "solvalues" vector
    solvalues[i] = (*matrix_state)[lid];
  }
  elevec1.putScalar(0.0);

  // EVALUATE SHAPE POLYNOMIALS IN NODE
  // In hdg we can have several more points inside the element than in the
  //"real" discretization and therefore it is necessary to compute the value
  // that the internal solution takes in the node of the discretization.

  // Cycling through all the "real" nodes of the element to get the coordinates
  // Remember that the coordinates are the local ones.
  for (unsigned int i = 0; i < nen_; ++i)
  {
    // Cycling through the spatial dimensions to get the coordinates
    for (unsigned int idim = 0; idim < nsd_; idim++) shapes_->xsi(idim) = locations(idim, i);

    // Evaluating the polinomials in the point given by "shapes_->xsi".
    // The polynomials are the internal ones.
    // The result of the evaluation is given in "values".
    shapes_->polySpace_->Evaluate(shapes_->xsi, values);

    // compute values for internal unknowns by summing over all basis functions
    for (unsigned int m = 0; m <= msd_; ++m)
    {
      double sum_mixedvar = 0.0;
      for (unsigned int k = 0; k < shapes_->ndofs_; ++k)
        sum_mixedvar += values(k) * solvalues[m * shapes_->ndofs_ + k];
      elevec1(m * nen_ + i) = sum_mixedvar;
    }
    {
      double sum_density = 0.0;
      for (unsigned int k = 0; k < shapes_->ndofs_; ++k)
        sum_density += values(k) * solvalues[msd_ * shapes_->ndofs_ + k];
      elevec1(msd_ * nen_ + i) = sum_density;
    }
  }

  // get trace solution values
  // Same as before bu this time the dimension is nsd_-1 because we went from
  // the interior to the faces. We have to be careful because we are using a
  // part of the previous vector. The coordinates are still in the local frame.
  locations = CORE::FE::getEleNodeNumbering_nodes_paramspace(
      CORE::FE::DisTypeToFaceShapeType<distype>::shape);

  // Storing the number of nodes for each face of the element as vector
  // NumberCornerNodes
  std::vector<int> ncn = CORE::FE::getNumberOfFaceElementCornerNodes(distype);
  // NumberInternalNodes
  std::vector<int> nin = CORE::FE::getNumberOfFaceElementInternalNodes(distype);

  // Now the vector "matrix_state" contains the trace velocity values following
  // the local id numbers
  matrix_state = discretization.GetState(0, "velnp");

  // we have always two dofsets
  Element::LocationArray la(2);
  ele->LocationVector(discretization, la, false);
  localDofs = la[0].lm_;
  solvalues.resize(localDofs.size());

  for (unsigned int i = 0; i < solvalues.size(); ++i)
  {
    const int lid = matrix_state->Map().LID(localDofs[i]);
    solvalues[i] = (*matrix_state)[lid];
  }
  for (int i = (msd_ + 1 + nsd_) * nen_; i < elevec1.numRows(); ++i) elevec1(i) = 0.0;

  CORE::LINALG::SerialDenseVector fvalues(shapesface_->nfdofs_);
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // Checking how many nodes the face has
    const int nfn = CORE::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace;

    // As already said, the dimension of the coordinate matrix is now nsd_-1
    // times the number of nodes in the face.
    CORE::LINALG::Matrix<nsd_ - 1, nfn> xsishuffle(true);

    // Cycling throught the nodes of the face to store the node positions in the
    // correct order using xsishuffle as a temporary vector
    for (int i = 0; i < nfn; ++i)
    {
      // cycling through the spatial dimensions
      for (unsigned int idim = 0; idim < nsd_ - 1; idim++)
      {
        // If the face belongs to the element being considered
        if (ele->Faces()[f]->ParentMasterElement() == ele)
          xsishuffle(idim, i) = locations(idim, i);
        else
          // If the face does not belong to the element being considered it is
          // necessary to change the ordering
          xsishuffle(idim, ele->Faces()[f]->GetLocalTrafoMap()[i]) = locations(idim, i);
      }
    }

    // EVALUATE SHAPE POLYNOMIALS IN NODE
    // Now that we have an ordered coordinates vector we can easily compute the
    // values of the shape functions in the nodes.
    for (int i = 0; i < nfn; ++i)
    {
      // Storing the actual coordinates of the current node
      for (unsigned int idim = 0; idim < nsd_ - 1; idim++)
        shapesface_->xsi(idim) = xsishuffle(idim, i);
      // Actually evaluating shape polynomials in node
      shapesface_->polySpace_->Evaluate(shapesface_->xsi, fvalues);

      // compute values for the trace by summing over all basis functions
      {
        double sum_trace_density = 0.0;
        for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
          sum_trace_density += fvalues(k) * solvalues[f * shapesface_->nfdofs_ * (1 + nsd_) + k];
        if (i < ncn[f])
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_density / nsd_;
        else if (i < nfn - nin[f])
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_density / (nsd_ - 1);
        else
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) += sum_trace_density;
      }
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        double sum_trace_momentum = 0.0;
        for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
        {
          sum_trace_momentum +=
              fvalues(k) *
              solvalues[f * shapesface_->nfdofs_ * (1 + nsd_) + (1 + d) * shapesface_->nfdofs_ + k];
        }
        if (i < ncn[f])
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum / nsd_;
        else if (i < nfn - nin[f])
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum / (nsd_ - 1);
        else
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum;
      }
    }
  }

  return 0;
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::EvaluateAll(const int funcnum,
    const CORE::LINALG::Matrix<nsd_, 1>& xyz, const double t, CORE::LINALG::Matrix<msd_, 1>& L,
    double& r, CORE::LINALG::Matrix<nsd_, 1>& w) const
{
  r = GLOBAL::Problem::Instance()
          ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(funcnum - 1)
          .Evaluate(xyz.A(), t, 0);

  for (unsigned int d = 0; d < nsd_; ++d)
    w(d) = GLOBAL::Problem::Instance()
               ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(funcnum - 1)
               .Evaluate(xyz.A(), t, 1 + d);

  for (unsigned int m = 0; m < msd_; ++m)
    L(m) = GLOBAL::Problem::Instance()
               ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(funcnum - 1)
               .Evaluate(xyz.A(), t, 1 + nsd_ + m);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::EvaluateDensityMomentum(const int funcnum,
    const CORE::LINALG::Matrix<nsd_, 1>& xyz, const double t, double& r,
    CORE::LINALG::Matrix<nsd_, 1>& w) const
{
  r = GLOBAL::Problem::Instance()
          ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(funcnum - 1)
          .Evaluate(xyz.A(), t, 0);

  for (unsigned int d = 0; d < nsd_; ++d)
    w(d) = GLOBAL::Problem::Instance()
               ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(funcnum - 1)
               .Evaluate(xyz.A(), t, 1 + d);
}



template <CORE::FE::CellType distype>
DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>*
DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::Instance(CORE::UTILS::SingletonAction action)
{
  static auto singleton_owner = CORE::UTILS::MakeSingletonOwner(
      []()
      {
        return std::unique_ptr<DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>>(
            new DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>());
      });

  return singleton_owner.Instance(action);
}



template <CORE::FE::CellType distype>
DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::LocalSolver(
    const DRT::ELEMENTS::Fluid* ele, const CORE::FE::ShapeValues<distype>& shapeValues,
    CORE::FE::ShapeValuesFace<distype>& shapeValuesFace, bool completepoly)
    : ndofs_(shapeValues.ndofs_),
      convective(true),
      unsteady(true),
      ale(ele->IsAle()),
      shapes_(shapeValues),
      shapesface_(shapeValuesFace)
{
  // count ndofs on faces
  ndofsfaces_ = 0;
  for (unsigned int i = 0; i < nfaces_; ++i)
  {
    shapesface_.EvaluateFace(*ele, i);
    ndofsfaces_ += shapesface_.nfdofs_;
  }

  // initialize stabilization parameters
  tau_r = 0.0;
  tau_w = 0.0;

  // initialize auxiliary matrices
  massPart.shape(ndofs_, shapes_.nqpoints_);
  massPartW.shape(ndofs_, shapes_.nqpoints_);
  massMat.shape(ndofs_, ndofs_);

  // initialize unknowns
  Leg.shape(msd_, shapes_.nqpoints_);
  reg.size(shapes_.nqpoints_);
  weg.shape(nsd_, shapes_.nqpoints_);

  // initialize ALE variables
  aeg.shape(nsd_, shapes_.nqpoints_);
  dadxyzeg.shape(nsd_ * nsd_, shapes_.nqpoints_);

  // initialize matrices
  ALL.shape(msd_ * ndofs_, msd_ * ndofs_);
  ALr.shape(msd_ * ndofs_, 1 * ndofs_);
  ALw.shape(msd_ * ndofs_, nsd_ * ndofs_);
  ALR.shape(msd_ * ndofs_, 1 * ndofsfaces_);
  ALW.shape(msd_ * ndofs_, nsd_ * ndofsfaces_);
  Arr.shape(1 * ndofs_, 1 * ndofs_);
  Arw.shape(1 * ndofs_, nsd_ * ndofs_);
  ArR.shape(1 * ndofs_, 1 * ndofsfaces_);
  ArW.shape(1 * ndofs_, nsd_ * ndofsfaces_);
  AwL.shape(nsd_ * ndofs_, msd_ * ndofs_);
  Awr.shape(nsd_ * ndofs_, 1 * ndofs_);
  Aww.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  AwR.shape(nsd_ * ndofs_, 1 * ndofsfaces_);
  AwW.shape(nsd_ * ndofs_, nsd_ * ndofsfaces_);
  ARr.shape(1 * ndofsfaces_, 1 * ndofs_);
  ARR.shape(1 * ndofsfaces_, 1 * ndofsfaces_);
  AWL.shape(nsd_ * ndofsfaces_, msd_ * ndofs_);
  AWw.shape(nsd_ * ndofsfaces_, nsd_ * ndofs_);
  AWR.shape(nsd_ * ndofsfaces_, 1 * ndofsfaces_);
  AWW.shape(nsd_ * ndofsfaces_, nsd_ * ndofsfaces_);

  // initialize residuals
  RL.size(msd_ * ndofs_);
  Rr.size(1 * ndofs_);
  Rw.size(nsd_ * ndofs_);
  RR.size(1 * ndofsfaces_);
  RW.size(nsd_ * ndofsfaces_);

  // initialize local/global matrices/vectors
  Klocallocal.shape((msd_ + 1 + nsd_) * ndofs_, (msd_ + 1 + nsd_) * ndofs_);
  Klocalglobal.shape((msd_ + 1 + nsd_) * ndofs_, (1 + nsd_) * ndofsfaces_);
  Kgloballocal.shape((1 + nsd_) * ndofsfaces_, (msd_ + 1 + nsd_) * ndofs_);
  Kglobalglobal.shape((1 + nsd_) * ndofsfaces_, (1 + nsd_) * ndofsfaces_);
  Rlocal.size((msd_ + 1 + nsd_) * ndofs_);
  Rglobal.size((1 + nsd_) * ndofsfaces_);
  KlocallocalInv.shape((msd_ + 1 + nsd_) * ndofs_, (msd_ + 1 + nsd_) * ndofs_);

  // pair of indices in Voigt notation
  int s = 0;
  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = d + 1; e < nsd_; ++e)
    {
      VoigtP[s][0] = d;
      VoigtP[s][1] = e;
      s = s + 1;
    }

  // pointer to class FluidEleParameter (access to the general parameter)
  fldparatimint_ = Teuchos::rcp(DRT::ELEMENTS::FluidEleParameterTimInt::Instance(), false);

  // initialize also general parameter list, also it will be overwritten in derived subclasses
  fldpara_ = Teuchos::rcp(DRT::ELEMENTS::FluidEleParameterStd::Instance(), false);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::InitializeAll()
{
  // initialize unknowns
  Leg.putScalar(0.0);
  reg.putScalar(0.0);
  weg.putScalar(0.0);

  // initialize ALE variables
  aeg.putScalar(0.0);
  dadxyzeg.putScalar(0.0);

  // initialize matrices
  ALL.putScalar(0.0);
  ALr.putScalar(0.0);
  ALw.putScalar(0.0);
  ALR.putScalar(0.0);
  ALW.putScalar(0.0);
  Arr.putScalar(0.0);
  Arw.putScalar(0.0);
  ArR.putScalar(0.0);
  ArW.putScalar(0.0);
  AwL.putScalar(0.0);
  Awr.putScalar(0.0);
  Aww.putScalar(0.0);
  AwR.putScalar(0.0);
  AwW.putScalar(0.0);
  ARr.putScalar(0.0);
  ARR.putScalar(0.0);
  AWL.putScalar(0.0);
  AWw.putScalar(0.0);
  AWR.putScalar(0.0);
  AWW.putScalar(0.0);

  // initialize residuals
  RL.putScalar(0.0);
  Rr.putScalar(0.0);
  Rw.putScalar(0.0);
  RR.putScalar(0.0);
  RW.putScalar(0.0);

  // initialize local/global matrices/vectors
  Klocallocal.putScalar(0.0);
  Klocalglobal.putScalar(0.0);
  Kgloballocal.putScalar(0.0);
  Kglobalglobal.putScalar(0.0);
  Rlocal.putScalar(0.0);
  Rglobal.putScalar(0.0);
  KlocallocalInv.putScalar(0.0);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeMaterialMatrix(
    const Teuchos::RCP<MAT::Material>& mat, const CORE::LINALG::Matrix<nsd_, 1>& xyz,
    CORE::LINALG::SerialDenseMatrix& DL, CORE::LINALG::SerialDenseMatrix& Dw)
{
  // initialize DL and Dw
  DL.shape(msd_, msd_);
  Dw.shape(msd_, msd_);

  // evaluate D_fac
  CORE::LINALG::SerialDenseMatrix D_fac(msd_, msd_);
  for (unsigned int m = 0; m < msd_; ++m) D_fac(m, m) = 1.0;
  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = 0; e < nsd_; ++e) D_fac(d, e) -= 1.0 / 3.0;

  // check for variable viscosity
  const Teuchos::ParameterList& fluidparams = GLOBAL::Problem::Instance()->FluidDynamicParams();
  int varviscfuncnum = fluidparams.get<int>("VARVISCFUNCNO");

  // compute material matrix
  if (varviscfuncnum < 0)  // constant viscosity
  {
    // get material
    const MAT::WeaklyCompressibleFluid* actmat =
        static_cast<const MAT::WeaklyCompressibleFluid*>(mat.get());

    // get viscosity
    double mu = actmat->Viscosity();

    // evaluate Dw
    for (unsigned int d = 0; d < nsd_; ++d) Dw(d, d) = std::pow(2.0 * mu, 1.0 / 2.0);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      Dw(nsd_ + s, nsd_ + s) = std::pow(mu, 1.0 / 2.0);

    // evaluate DL
    CORE::LINALG::multiply(DL, Dw, D_fac);
  }
  else  // variable viscosity
  {
    // get time
    const double time = fldparatimint_->Time();

    // get viscosity
    double mu = GLOBAL::Problem::Instance()
                    ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(varviscfuncnum - 1)
                    .Evaluate(xyz.A(), time, 0);

    // evaluate Dw
    for (unsigned int d = 0; d < nsd_; ++d) Dw(d, d) = 2.0 * mu;
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s) Dw(nsd_ + s, nsd_ + s) = mu;

    // evaluate DL
    DL = D_fac;
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeInteriorResidual(
    const Teuchos::RCP<MAT::Material>& mat, const std::vector<double>& val,
    const std::vector<double>& accel, const std::vector<double>& alevel)
{
  // set convective flag
  INPAR::FLUID::PhysicalType physicaltype = fldpara_->PhysicalType();
  convective = (physicaltype != INPAR::FLUID::weakly_compressible_stokes_dens_mom);

  // set unsteady flag
  const Teuchos::ParameterList& fluidparams = GLOBAL::Problem::Instance()->FluidDynamicParams();
  std::string timeintegr = fluidparams.get<std::string>("TIMEINTEGR");
  unsteady = (timeintegr.compare("Stationary") != 0);

  // get material properties
  const MAT::WeaklyCompressibleFluid* actmat =
      static_cast<const MAT::WeaklyCompressibleFluid*>(mat.get());

  // get force function
  int forcefuncnum = fluidparams.get<int>("BODYFORCEFUNCNO");

  // get time
  const double time = fldparatimint_->Time();

  // ease notation
  CORE::LINALG::SerialDenseMatrix N = shapes_.shfunct;
  CORE::LINALG::SerialDenseMatrix Nn = shapes_.funct;
  CORE::LINALG::SerialDenseMatrix Nxyz = shapes_.shderxy;
  CORE::LINALG::SerialDenseMatrix Nnxyz = shapes_.derxy;
  CORE::LINALG::SerialDenseVector fac = shapes_.jfac;
  unsigned int nqpoints = shapes_.nqpoints_;

  // extract interior nodal values
  CORE::LINALG::Matrix<nsd_, 1> xyze;
  CORE::LINALG::SerialDenseMatrix Le(msd_, ndofs_);
  CORE::LINALG::SerialDenseVector re(ndofs_);
  CORE::LINALG::SerialDenseMatrix we(nsd_, ndofs_);
  CORE::LINALG::SerialDenseVector pe(ndofs_);
  CORE::LINALG::SerialDenseVector drdte(ndofs_);
  CORE::LINALG::SerialDenseMatrix dwdte(nsd_, ndofs_);
  CORE::LINALG::SerialDenseMatrix DLe(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dwe(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix DwLe(msd_, ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d) = shapes_.nodexyzreal(d, i);

    for (unsigned int m = 0; m < msd_; ++m) Le(m, i) = val[m * ndofs_ + i];

    re(i) = val[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) we(d, i) = val[(msd_ + 1 + d) * ndofs_ + i];

    pe(i) = actmat->ComputePressure(re[i]);

    drdte(i) = accel[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) dwdte(d, i) = accel[(msd_ + 1 + d) * ndofs_ + i];

    // compute material matrix
    ComputeMaterialMatrix(mat, xyze, DLe, Dwe);

    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int e = 0; e < nsd_; ++e) DwLe(d, i) += Dwe(d, e) * Le(e, i);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      DwLe(nsd_ + s, i) += Dwe(nsd_ + s, nsd_ + s) * Le(nsd_ + s, i);
  }

  // extract ale nodal values
  CORE::LINALG::SerialDenseMatrix ae(nsd_, ndofs_);

  if (ale)
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) ae(d, n) = alevel[n * nsd_ + d];

  // initialize values interpolated on gauss points
  CORE::LINALG::Matrix<nsd_, 1> xyzeg;
  CORE::LINALG::SerialDenseMatrix feg(1 + nsd_, nqpoints);
  CORE::LINALG::SerialDenseVector drdteg(nqpoints);
  CORE::LINALG::SerialDenseMatrix dwdteg(nsd_, nqpoints);
  CORE::LINALG::SerialDenseMatrix dDwLdxyzeg(msd_ * nsd_, nqpoints);
  CORE::LINALG::SerialDenseMatrix dpdxyzeg(1 * nsd_, nqpoints);
  CORE::LINALG::SerialDenseMatrix DLeg(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dweg(msd_, msd_);

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d) xyzeg(d) = shapes_.xyzreal(d, q);

      for (unsigned int m = 0; m < msd_; ++m) Leg(m, q) += N(i, q) * Le(m, i);

      reg(q) += N(i, q) * re(i);

      for (unsigned int d = 0; d < nsd_; ++d) weg(d, q) += N(i, q) * we(d, i);

      if (forcefuncnum > 0)
        for (unsigned int dmod = 0; dmod < (1 + nsd_); ++dmod)
          feg(dmod, q) = GLOBAL::Problem::Instance()
                             ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(forcefuncnum - 1)
                             .Evaluate(xyzeg.A(), time, dmod);

      drdteg(q) += N(i, q) * drdte(i);

      for (unsigned int d = 0; d < nsd_; ++d) dwdteg(d, q) += N(i, q) * dwdte(d, i);

      for (unsigned int m = 0; m < msd_; ++m)
        for (unsigned int d = 0; d < nsd_; ++d)
          dDwLdxyzeg(m * nsd_ + d, q) += Nxyz(i * nsd_ + d, q) * DwLe(m, i);

      for (unsigned int d = 0; d < nsd_; ++d) dpdxyzeg(d, q) += Nxyz(i * nsd_ + d, q) * pe(i);
    }

    // compute material matrix
    ComputeMaterialMatrix(mat, xyzeg, DLeg, Dweg);

    // interpolate ale values on gauss points
    if (ale)
      for (unsigned int n = 0; n < nen_; ++n)
      {
        for (unsigned int d = 0; d < nsd_; ++d) aeg(d, q) += Nn(n, q) * ae(d, n);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            dadxyzeg(d * nsd_ + e, q) += Nnxyz(n * nsd_ + e, q) * ae(d, n);
      }

    // compute residuals
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) RL(m * ndofs_ + i) += +N(i, q) * Leg(m, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RL(d * ndofs_ + i) += -Nxyz(i * nsd_ + e, q) * DLeg(e, d) * weg(e, q) / reg(q) * fac(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        RL((nsd_ + s) * ndofs_ + i) += -Nxyz(i * nsd_ + VoigtP[s][0], q) *
                                           DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][1], q) /
                                           reg(q) * fac(q) -
                                       Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) *
                                           weg(VoigtP[s][0], q) / reg(q) * fac(q);

      if (unsteady) Rr(i) += -N(i, q) * drdteg(q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += -N(i, q) * reg(q) * dadxyzeg(d * nsd_ + d, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d) Rr(i) += +Nxyz(i * nsd_ + d, q) * weg(d, q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += -Nxyz(i * nsd_ + d, q) * reg(q) * aeg(d, q) * fac(q);

      Rr(i) += +N(i, q) * feg(0, q) * fac(q);

      if (unsteady)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rw(d * ndofs_ + i) += -N(i, q) * dwdteg(d, q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += -N(i, q) * weg(d, q) * dadxyzeg(e * nsd_ + e, q) * fac(q);

      if (convective)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += +Nxyz(i * nsd_ + e, q) * weg(d, q) * weg(e, q) / reg(q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += -Nxyz(i * nsd_ + e, q) * weg(d, q) * aeg(e, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -N(i, q) * dDwLdxyzeg(d * nsd_ + d, q) * fac(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      {
        Rw(VoigtP[s][0] * ndofs_ + i) +=
            -N(i, q) * dDwLdxyzeg((nsd_ + s) * nsd_ + VoigtP[s][1], q) * fac(q);
        Rw(VoigtP[s][1] * ndofs_ + i) +=
            -N(i, q) * dDwLdxyzeg((nsd_ + s) * nsd_ + VoigtP[s][0], q) * fac(q);
      }

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -N(i, q) * dpdxyzeg(d, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += +N(i, q) * feg(1 + d, q) * fac(q);
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeInteriorMatrices(
    const Teuchos::RCP<MAT::Material>& mat)
{
  // get material properties
  const MAT::WeaklyCompressibleFluid* actmat =
      static_cast<const MAT::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->ComprCoeff();

  // inverse of time factor
  const double invtimefac = 1.0 / (fldparatimint_->TimeFac());

  // ease notation
  CORE::LINALG::SerialDenseMatrix N = shapes_.shfunct;
  CORE::LINALG::SerialDenseMatrix Nxyz = shapes_.shderxy;
  CORE::LINALG::SerialDenseVector fac = shapes_.jfac;
  unsigned int nqpoints = shapes_.nqpoints_;

  // extract interior nodal values
  CORE::LINALG::Matrix<nsd_, 1> xyze;
  CORE::LINALG::SerialDenseMatrix DLe(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dwe(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dwemod(nsd_ * nsd_ + (msd_ - nsd_), ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d) = shapes_.nodexyzreal(d, i);

    // compute material matrix
    ComputeMaterialMatrix(mat, xyze, DLe, Dwe);

    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int e = 0; e < nsd_; ++e) Dwemod(d * nsd_ + e, i) = Dwe(d, e);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      Dwemod(nsd_ * nsd_ + s, i) = Dwe(nsd_ + s, nsd_ + s);
  }

  // initialize values interpolated on gauss points
  CORE::LINALG::Matrix<nsd_, 1> xyzeg;
  CORE::LINALG::SerialDenseMatrix DLeg(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dweg(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix dDwdxyzeg((nsd_ * nsd_ + (msd_ - nsd_)) * nsd_, nqpoints);

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzeg(d) = shapes_.xyzreal(d, q);

    // compute material matrix
    ComputeMaterialMatrix(mat, xyzeg, DLeg, Dweg);

    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int mmod = 0; mmod < (nsd_ * nsd_ + (msd_ - nsd_)); ++mmod)
        for (unsigned int d = 0; d < nsd_; ++d)
          dDwdxyzeg(mmod * nsd_ + d, q) += Nxyz(i * nsd_ + d, q) * Dwemod(mmod, i);
    }

    // compute matrices
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        for (unsigned int m = 0; m < msd_; ++m)
          ALL(m * ndofs_ + i, m * ndofs_ + j) += -N(i, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALr(d * ndofs_ + i, j) += -Nxyz(i * nsd_ + e, q) * DLeg(e, d) * weg(e, q) /
                                      std::pow(reg(q), 2.0) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
          ALr((nsd_ + s) * ndofs_ + i, j) +=
              -Nxyz(i * nsd_ + VoigtP[s][0], q) * DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][1], q) /
                  std::pow(reg(q), 2.0) * N(j, q) * fac(q) -
              Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][0], q) /
                  std::pow(reg(q), 2.0) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALw(d * ndofs_ + i, e * ndofs_ + j) +=
                +Nxyz(i * nsd_ + e, q) * DLeg(e, d) / reg(q) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          ALw((nsd_ + s) * ndofs_ + i, VoigtP[s][0] * ndofs_ + j) +=
              +Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) / reg(q) * N(j, q) *
              fac(q);
          ALw((nsd_ + s) * ndofs_ + i, VoigtP[s][1] * ndofs_ + j) +=
              +Nxyz(i * nsd_ + VoigtP[s][0], q) * DLeg(nsd_ + s, nsd_ + s) / reg(q) * N(j, q) *
              fac(q);
        }

        if (unsteady) Arr(i, j) += +invtimefac * N(i, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            Arr(i, j) += +N(i, q) * dadxyzeg(d * nsd_ + d, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            Arr(i, j) += +Nxyz(i * nsd_ + d, q) * aeg(d, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Arw(i, d * ndofs_ + j) += -Nxyz(i * nsd_ + d, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AwL(d * ndofs_ + i, e * ndofs_ + j) +=
                +N(i, q) * Dweg(d, e) * Nxyz(j * nsd_ + d, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AwL(VoigtP[s][0] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * Dweg(nsd_ + s, nsd_ + s) * Nxyz(j * nsd_ + VoigtP[s][1], q) * fac(q);
          AwL(VoigtP[s][1] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * Dweg(nsd_ + s, nsd_ + s) * Nxyz(j * nsd_ + VoigtP[s][0], q) * fac(q);
        }

        // TODO to test this term. It arises from the linearization of the variable viscosity term
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AwL(d * ndofs_ + i, e * ndofs_ + j) +=
                +N(i, q) * dDwdxyzeg((d * nsd_ + e) * nsd_ + d, q) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AwL(VoigtP[s][0] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * dDwdxyzeg((nsd_ * nsd_ + s) * nsd_ + VoigtP[s][1], q) * N(j, q) * fac(q);
          AwL(VoigtP[s][1] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * dDwdxyzeg((nsd_ * nsd_ + s) * nsd_ + VoigtP[s][0], q) * N(j, q) * fac(q);
        }

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Awr(d * ndofs_ + i, j) += +Nxyz(i * nsd_ + e, q) * weg(d, q) * weg(e, q) /
                                        std::pow(reg(q), 2.0) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Awr(d * ndofs_ + i, j) += +1.0 / eps * N(i, q) * Nxyz(j * nsd_ + d, q) * fac(q);

        if (unsteady)
          for (unsigned int d = 0; d < nsd_; ++d)
            Aww(d * ndofs_ + i, d * ndofs_ + j) += +invtimefac * N(i, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  +N(i, q) * dadxyzeg(e * nsd_ + e, q) * N(j, q) * fac(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  -Nxyz(i * nsd_ + e, q) * weg(e, q) / reg(q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  +Nxyz(i * nsd_ + e, q) * aeg(e, q) * N(j, q) * fac(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, e * ndofs_ + j) +=
                  -Nxyz(i * nsd_ + e, q) * weg(d, q) / reg(q) * N(j, q) * fac(q);
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeFaceResidual(const int f,
    const Teuchos::RCP<MAT::Material>& mat, const std::vector<double>& val,
    const std::vector<double>& traceval, const std::vector<double>& alevel)
{
  // get material properties
  const MAT::WeaklyCompressibleFluid* actmat =
      static_cast<const MAT::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->ComprCoeff();

  // get stabilization parameters
  const Teuchos::ParameterList& fluidparams = GLOBAL::Problem::Instance()->FluidDynamicParams();
  double tau_r_ref = fluidparams.get<double>("STAB_DEN_REF");
  double tau_w_ref = fluidparams.get<double>("STAB_MOM_REF");
  tau_r = tau_r_ref / eps;
  tau_w = tau_w_ref;

  // ease notation
  CORE::FE::ShapeValuesInteriorOnFace Ni = shapesface_.shfunctI;
  CORE::LINALG::SerialDenseMatrix Nf = shapesface_.shfunct;
  CORE::LINALG::SerialDenseMatrix Nn = shapesface_.funct;
  CORE::LINALG::SerialDenseMatrix nxyz = shapesface_.normals;
  CORE::LINALG::SerialDenseVector facf = shapesface_.jfac;
  unsigned int nfqpoints = shapesface_.nqpoints_;
  unsigned int nfdofs = shapesface_.nfdofs_;
  unsigned int nfn = shapesface_.nfn_;
  std::vector<std::vector<int>> faceNodeOrder = shapesface_.faceNodeOrder;

  // extract interior nodal values
  CORE::LINALG::SerialDenseMatrix Le(msd_, ndofs_);
  CORE::LINALG::SerialDenseVector re(ndofs_);
  CORE::LINALG::SerialDenseMatrix we(nsd_, ndofs_);
  CORE::LINALG::SerialDenseVector pe(ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int m = 0; m < msd_; ++m) Le(m, i) = val[m * ndofs_ + i];

    re(i) = val[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) we(d, i) = val[(msd_ + 1 + d) * ndofs_ + i];

    pe(i) = actmat->ComputePressure(re[i]);
  }

  // extract ale nodal values
  CORE::LINALG::SerialDenseMatrix aef(nsd_, nfn);

  if (ale)
    for (unsigned int n = 0; n < nfn; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) aef(d, n) = alevel[faceNodeOrder[f][n] * nsd_ + d];

  // extract trace nodal value
  CORE::LINALG::SerialDenseVector rhatef(nfdofs);
  CORE::LINALG::SerialDenseMatrix whatef(nsd_, nfdofs);
  for (unsigned int i = 0; i < nfdofs; ++i)
  {
    rhatef(i) = traceval[f * nfdofs * (1 + nsd_) + i];

    for (unsigned int d = 0; d < nsd_; ++d)
      whatef(d, i) = traceval[f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i];
  }

  // initialize values interpolated on gauss points
  CORE::LINALG::Matrix<nsd_, 1> xyzefg;
  CORE::LINALG::SerialDenseMatrix Lefg(msd_, nfqpoints);
  CORE::LINALG::SerialDenseVector refg(nfqpoints);
  CORE::LINALG::SerialDenseMatrix wefg(nsd_, nfqpoints);
  CORE::LINALG::SerialDenseVector phatefg(nfqpoints);
  CORE::LINALG::SerialDenseMatrix DLefg(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dwefg(msd_, msd_);
  rhatefg.size(nfqpoints);
  whatefg.shape(nsd_, nfqpoints);
  aefg.shape(nsd_, nfqpoints);

  // loop over quadrature points
  for (unsigned int q = 0; q < nfqpoints; ++q)
  {
    // interpolate interior values on gauss points
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) Lefg(m, q) += Ni(i, q) * Le(m, i);

      refg(q) += Ni(i, q) * re(i);

      for (unsigned int d = 0; d < nsd_; ++d) wefg(d, q) += Ni(i, q) * we(d, i);
    }

    // interpolate ale values on gauss points
    if (ale)
      for (unsigned int n = 0; n < nfn; ++n)
        for (unsigned int d = 0; d < nsd_; ++d) aefg(d, q) += Nn(n, q) * aef(d, n);

    // interpolate trace values on gauss points
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      rhatefg(q) += Nf(i, q) * rhatef(i);

      for (unsigned int d = 0; d < nsd_; ++d) whatefg(d, q) += Nf(i, q) * whatef(d, i);

      phatefg(q) = actmat->ComputePressure(rhatefg(q));
    }

    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzefg(d) = shapesface_.xyzreal(d, q);

    // compute material matrix
    ComputeMaterialMatrix(mat, xyzefg, DLefg, Dwefg);

    // compute residuals (interior contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RL(d * ndofs_ + i) +=
              +Ni(i, q) * nxyz(e, q) * DLefg(e, d) * whatefg(e, q) / rhatefg(q) * facf(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        RL((nsd_ + s) * ndofs_ + i) +=
            +Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) *
                whatefg(VoigtP[s][1], q) / rhatefg(q) * facf(q) +
            Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) *
                whatefg(VoigtP[s][0], q) / rhatefg(q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rr(i) += -Ni(i, q) * whatefg(d, q) * nxyz(d, q) * facf(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += +Ni(i, q) * rhatefg(q) * aefg(d, q) * nxyz(d, q) * facf(q);

      Rr(i) += -Ni(i, q) * tau_r * refg(q) * facf(q);

      Rr(i) += +Ni(i, q) * tau_r * rhatefg(q) * facf(q);

      if (convective)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) +=
                -Ni(i, q) * whatefg(d, q) * whatefg(e, q) / rhatefg(q) * nxyz(e, q) * facf(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += +Ni(i, q) * whatefg(d, q) * aefg(e, q) * nxyz(e, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -Ni(i, q) * tau_w * wefg(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += +Ni(i, q) * tau_w * whatefg(d, q) * facf(q);
    }

    // compute residuals (face contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      RR(f * nfdofs + i) += -Nf(i, q) * tau_r * refg(q) * facf(q);

      RR(f * nfdofs + i) += +Nf(i, q) * tau_r * rhatefg(q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RW((f * nsd_ + d) * nfdofs + i) +=
              -Nf(i, q) * Dwefg(d, e) * Lefg(e, q) * nxyz(d, q) * facf(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      {
        RW((f * nsd_ + VoigtP[s][0]) * nfdofs + i) += -Nf(i, q) * Dwefg(nsd_ + s, nsd_ + s) *
                                                      Lefg(nsd_ + s, q) * nxyz(VoigtP[s][1], q) *
                                                      facf(q);
        RW((f * nsd_ + VoigtP[s][1]) * nfdofs + i) += -Nf(i, q) * Dwefg(nsd_ + s, nsd_ + s) *
                                                      Lefg(nsd_ + s, q) * nxyz(VoigtP[s][0], q) *
                                                      facf(q);
      }

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += -Nf(i, q) * phatefg(q) * nxyz(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += -Nf(i, q) * tau_w * wefg(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += +Nf(i, q) * tau_w * whatefg(d, q) * facf(q);

      //  TODO Neumann term
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeFaceMatrices(
    const int f, const Teuchos::RCP<MAT::Material>& mat)
{
  // get material properties
  const MAT::WeaklyCompressibleFluid* actmat =
      static_cast<const MAT::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->ComprCoeff();

  // ease notation
  CORE::FE::ShapeValuesInteriorOnFace Ni = shapesface_.shfunctI;
  CORE::LINALG::SerialDenseMatrix Nf = shapesface_.shfunct;
  CORE::LINALG::SerialDenseMatrix nxyz = shapesface_.normals;
  CORE::LINALG::SerialDenseVector facf = shapesface_.jfac;
  unsigned int nfqpoints = shapesface_.nqpoints_;
  unsigned int nfdofs = shapesface_.nfdofs_;

  // initialize values interpolated on gauss points
  CORE::LINALG::Matrix<nsd_, 1> xyzefg;
  CORE::LINALG::SerialDenseMatrix DLefg(msd_, msd_);
  CORE::LINALG::SerialDenseMatrix Dwefg(msd_, msd_);

  // loop over quadrature points
  for (unsigned int q = 0; q < nfqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzefg(d) = shapesface_.xyzreal(d, q);

    // compute material matrix
    ComputeMaterialMatrix(mat, xyzefg, DLefg, Dwefg);

    // compute matrices (interior - interior contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        Arr(i, j) += +Ni(i, q) * tau_r * Ni(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Aww(d * ndofs_ + i, d * ndofs_ + j) += +Ni(i, q) * tau_w * Ni(j, q) * facf(q);
      }
    }

    // compute matrices (interior - face contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALR(d * ndofs_ + i, f * nfdofs + j) += +Ni(i, q) * nxyz(e, q) * DLefg(e, d) *
                                                   whatefg(e, q) / std::pow(rhatefg(q), 2.0) *
                                                   Nf(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
          ALR((nsd_ + s) * ndofs_ + i, f * nfdofs + j) +=
              +Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) *
                  whatefg(VoigtP[s][1], q) / std::pow(rhatefg(q), 2.0) * Nf(j, q) * facf(q) +
              Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) *
                  whatefg(VoigtP[s][0], q) / std::pow(rhatefg(q), 2.0) * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j) +=
                -Ni(i, q) * nxyz(e, q) * DLefg(e, d) / rhatefg(q) * Nf(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          ALW((nsd_ + s) * ndofs_ + i, (f * nsd_ + VoigtP[s][0]) * nfdofs + j) +=
              -Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) / rhatefg(q) *
              Nf(j, q) * facf(q);
          ALW((nsd_ + s) * ndofs_ + i, (f * nsd_ + VoigtP[s][1]) * nfdofs + j) +=
              -Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) / rhatefg(q) *
              Nf(j, q) * facf(q);
        }

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            ArR(i, f * nfdofs + j) += -Ni(i, q) * aefg(d, q) * nxyz(d, q) * Nf(j, q) * facf(q);

        ArR(i, f * nfdofs + j) += -Ni(i, q) * tau_r * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          ArW(i, (f * nsd_ + d) * nfdofs + j) += +Ni(i, q) * nxyz(d, q) * Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwR(d * ndofs_ + i, f * nfdofs + j) += -Ni(i, q) * whatefg(d, q) * whatefg(e, q) /
                                                     std::pow(rhatefg(q), 2.0) * nxyz(e, q) *
                                                     Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
                  +Ni(i, q) * whatefg(e, q) / rhatefg(q) * nxyz(e, q) * Nf(j, q) * facf(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
                  -Ni(i, q) * aefg(e, q) * nxyz(e, q) * Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j) +=
                  +Ni(i, q) * whatefg(d, q) / rhatefg(q) * nxyz(e, q) * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
              -Ni(i, q) * tau_w * Nf(j, q) * facf(q);
      }
    }

    // compute matrices (face - interior contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        ARr(f * nfdofs + i, j) += +Nf(i, q) * tau_r * Ni(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AWL((f * nsd_ + d) * nfdofs + i, e * ndofs_ + j) +=
                +Nf(i, q) * nxyz(d, q) * Dwefg(d, e) * Ni(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AWL((f * nsd_ + VoigtP[s][0]) * nfdofs + i, (nsd_ + s) * ndofs_ + j) +=
              +Nf(i, q) * nxyz(VoigtP[s][1], q) * Dwefg(nsd_ + s, nsd_ + s) * Ni(j, q) * facf(q);
          AWL((f * nsd_ + VoigtP[s][1]) * nfdofs + i, (nsd_ + s) * ndofs_ + j) +=
              +Nf(i, q) * nxyz(VoigtP[s][0], q) * Dwefg(nsd_ + s, nsd_ + s) * Ni(j, q) * facf(q);
        }

        for (unsigned int d = 0; d < nsd_; ++d)
          AWw((f * nsd_ + d) * nfdofs + i, d * ndofs_ + j) +=
              +Nf(i, q) * tau_w * Ni(j, q) * facf(q);
      }
    }

    // compute matrices (face - face contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        ARR(f * nfdofs + i, f * nfdofs + j) += -Nf(i, q) * tau_r * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AWR((f * nsd_ + d) * nfdofs + i, f * nfdofs + j) +=
              +Nf(i, q) * nxyz(d, q) / eps * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AWW((f * nsd_ + d) * nfdofs + i, (f * nsd_ + d) * nfdofs + j) +=
              -Nf(i, q) * tau_w * Nf(j, q) * facf(q);
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeLocalResidual()
{
  // fill vector
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int m = 0; m < msd_; ++m) Rlocal(m * ndofs_ + i) = RL(m * ndofs_ + i);

    Rlocal(msd_ * ndofs_ + i) = Rr(i);

    for (unsigned int d = 0; d < nsd_; ++d)
      Rlocal((msd_ + 1 + d) * ndofs_ + i) = Rw(d * ndofs_ + i);
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeGlobalResidual(
    DRT::ELEMENTS::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.EvaluateFace(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill vector
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      Rglobal(f * nfdofs * (1 + nsd_) + i) = RR(f * nfdofs + i);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i) = RW((f * nsd_ + d) * nfdofs + i);
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeLocalLocalMatrix()
{
  // fill matrix
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int j = 0; j < ndofs_; ++j)
    {
      for (unsigned int m = 0; m < msd_; ++m)
      {
        for (unsigned int n = 0; n < msd_; ++n)
          Klocallocal(m * ndofs_ + i, n * ndofs_ + j) = ALL(m * ndofs_ + i, n * ndofs_ + j);

        Klocallocal(m * ndofs_ + i, msd_ * ndofs_ + j) = ALr(m * ndofs_ + i, j);

        for (unsigned int d = 0; d < nsd_; ++d)
          Klocallocal(m * ndofs_ + i, (msd_ + 1 + d) * ndofs_ + j) =
              ALw(m * ndofs_ + i, d * ndofs_ + j);
      }

      Klocallocal(msd_ * ndofs_ + i, msd_ * ndofs_ + j) = Arr(i, j);

      for (unsigned int d = 0; d < nsd_; ++d)
        Klocallocal(msd_ * ndofs_ + i, (msd_ + 1 + d) * ndofs_ + j) = Arw(i, d * ndofs_ + j);

      for (unsigned int d = 0; d < nsd_; ++d)
      {
        for (unsigned int m = 0; m < msd_; ++m)
          Klocallocal((msd_ + 1 + d) * ndofs_ + i, m * ndofs_ + j) =
              AwL(d * ndofs_ + i, m * ndofs_ + j);

        Klocallocal((msd_ + 1 + d) * ndofs_ + i, msd_ * ndofs_ + j) = Awr(d * ndofs_ + i, j);

        for (unsigned int e = 0; e < nsd_; ++e)
          Klocallocal((msd_ + 1 + d) * ndofs_ + i, (msd_ + 1 + e) * ndofs_ + j) =
              Aww(d * ndofs_ + i, e * ndofs_ + j);
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeLocalGlobalMatrix(
    DRT::ELEMENTS::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.EvaluateFace(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        for (unsigned int m = 0; m < msd_; ++m)
        {
          Klocalglobal(m * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) =
              ALR(m * ndofs_ + i, f * nfdofs + j);

          for (unsigned int d = 0; d < nsd_; ++d)
            Klocalglobal(m * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + j) =
                ALW(m * ndofs_ + i, (f * nsd_ + d) * nfdofs + j);
        }

        Klocalglobal(msd_ * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) = ArR(i, f * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
          Klocalglobal(msd_ * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + j) =
              ArW(i, (f * nsd_ + d) * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          Klocalglobal((msd_ + 1 + d) * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) =
              AwR(d * ndofs_ + i, f * nfdofs + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Klocalglobal(
                (msd_ + 1 + d) * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + e) * nfdofs + j) =
                AwW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j);
        }
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeGlobalLocalMatrix(
    DRT::ELEMENTS::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.EvaluateFace(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        Kgloballocal(f * nfdofs * (1 + nsd_) + i, msd_ * ndofs_ + j) = ARr(f * nfdofs + i, j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          for (unsigned int m = 0; m < msd_; ++m)
            Kgloballocal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i, m * ndofs_ + j) =
                AWL((f * nsd_ + d) * nfdofs + i, m * ndofs_ + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Kgloballocal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
                (msd_ + 1 + e) * ndofs_ + j) = AWw((f * nsd_ + d) * nfdofs + i, e * ndofs_ + j);
        }
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::ComputeGlobalGlobalMatrix(
    DRT::ELEMENTS::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.EvaluateFace(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        Kglobalglobal(f * nfdofs * (1 + nsd_) + i, f * nfdofs * (1 + nsd_) + j) =
            ARR(f * nfdofs + i, f * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          Kglobalglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
              f * nfdofs * (1 + nsd_) + j) = AWR((f * nsd_ + d) * nfdofs + i, f * nfdofs + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Kglobalglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
                f * nfdofs * (1 + nsd_) + (1 + e) * nfdofs + j) =
                AWW((f * nsd_ + d) * nfdofs + i, (f * nsd_ + e) * nfdofs + j);
        }
      }
    }
  }
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::InvertLocalLocalMatrix()
{
  KlocallocalInv = Klocallocal;
  KlocallocalInvSolver.setMatrix(Teuchos::rcpFromRef(KlocallocalInv));
  int err = KlocallocalInvSolver.invert();
  if (err != 0) FOUR_C_THROW("Inversion of local-local matrix failed with errorcode %d", err);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::CondenseLocalResidual(
    CORE::LINALG::SerialDenseVector& eleVec)
{
  // initialize element vector
  eleVec.size((1 + nsd_) * ndofsfaces_);

  // create auxiliary vector
  CORE::LINALG::SerialDenseVector eleVecAux;
  eleVecAux.size((msd_ + 1 + nsd_) * ndofs_);

  // compute element vector
  CORE::LINALG::multiply(eleVecAux, KlocallocalInv, Rlocal);
  eleVec = Rglobal;
  CORE::LINALG::multiply(1.0, eleVec, -1.0, Kgloballocal, eleVecAux);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::CondenseLocalMatrix(
    CORE::LINALG::SerialDenseMatrix& eleMat)
{
  // initialize element matrix
  eleMat.shape((1 + nsd_) * ndofsfaces_, (1 + nsd_) * ndofsfaces_);

  // create auxiliary matrix
  CORE::LINALG::SerialDenseMatrix eleMatAux;
  eleMatAux.shape((msd_ + 1 + nsd_) * ndofs_, (1 + nsd_) * ndofsfaces_);

  // compute element matrix
  CORE::LINALG::multiply(eleMatAux, KlocallocalInv, Klocalglobal);
  eleMat = Kglobalglobal;
  CORE::LINALG::multiply(1.0, eleMat, -1.0, Kgloballocal, eleMatAux);
}



template <CORE::FE::CellType distype>
void DRT::ELEMENTS::FluidEleCalcHDGWeakComp<distype>::LocalSolver::PrintMatricesAndResiduals(
    DRT::ELEMENTS::Fluid& ele, CORE::LINALG::SerialDenseVector& eleVec,
    CORE::LINALG::SerialDenseMatrix& eleMat)
{
  // element
  std::cout << "\n\n Element number = " << ele.Id() + 1;

  // matrices
  std::cout << "\n\n ALL = \n\n";
  ALL.print(std::cout);
  std::cout << "\n\n ALr = \n\n";
  ALr.print(std::cout);
  std::cout << "\n\n ALw = \n\n";
  ALw.print(std::cout);
  std::cout << "\n\n ALR = \n\n";
  ALR.print(std::cout);
  std::cout << "\n\n ALW = \n\n";
  ALW.print(std::cout);
  std::cout << "\n\n Arr = \n\n";
  Arr.print(std::cout);
  std::cout << "\n\n Arw = \n\n";
  Arw.print(std::cout);
  std::cout << "\n\n ArR = \n\n";
  ArR.print(std::cout);
  std::cout << "\n\n ArW = \n\n";
  ArW.print(std::cout);
  std::cout << "\n\n AwL = \n\n";
  AwL.print(std::cout);
  std::cout << "\n\n Awr = \n\n";
  Awr.print(std::cout);
  std::cout << "\n\n Aww = \n\n";
  Aww.print(std::cout);
  std::cout << "\n\n AwR = \n\n";
  AwR.print(std::cout);
  std::cout << "\n\n AwW = \n\n";
  AwW.print(std::cout);
  std::cout << "\n\n ARr = \n\n";
  ARr.print(std::cout);
  std::cout << "\n\n ARR = \n\n";
  ARR.print(std::cout);
  std::cout << "\n\n AWL = \n\n";
  AWL.print(std::cout);
  std::cout << "\n\n AWw = \n\n";
  AWw.print(std::cout);
  std::cout << "\n\n AWR = \n\n";
  AWR.print(std::cout);
  std::cout << "\n\n AWW = \n\n";
  AWW.print(std::cout);

  // residuals
  std::cout << "\n\n RL = \n\n";
  RL.print(std::cout);
  std::cout << "\n\n Rr = \n\n";
  Rr.print(std::cout);
  std::cout << "\n\n Rw = \n\n";
  Rw.print(std::cout);
  std::cout << "\n\n RR = \n\n";
  RR.print(std::cout);
  std::cout << "\n\n RW = \n\n";
  RW.print(std::cout);

  // local/global matrices/vectors
  std::cout << "\n\n Klocallocal = \n\n";
  Klocallocal.print(std::cout);
  std::cout << "\n\n Klocalglobal = \n\n";
  Klocalglobal.print(std::cout);
  std::cout << "\n\n Kgloballocal = \n\n";
  Kgloballocal.print(std::cout);
  std::cout << "\n\n Kglobalglobal = \n\n";
  Kglobalglobal.print(std::cout);
  std::cout << "\n\n Rlocal = \n\n";
  Rlocal.print(std::cout);
  std::cout << "\n\n Rglobal = \n\n";
  Rglobal.print(std::cout);
  std::cout << "\n\n KlocallocalInv = \n\n";
  KlocallocalInv.print(std::cout);

  // element vector and matrix
  std::cout << "\n\n eleVec = \n\n";
  eleVec.print(std::cout);
  std::cout << "\n\n eleMat = \n\n";
  eleMat.print(std::cout);
}



// explicit instantiation of template classes
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::hex8>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::hex20>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::hex27>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::tet4>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::tet10>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::wedge6>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::wedge15>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::pyramid5>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::quad4>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::quad8>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::quad9>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::tri3>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::tri6>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::nurbs9>;
template class DRT::ELEMENTS::FluidEleCalcHDGWeakComp<CORE::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
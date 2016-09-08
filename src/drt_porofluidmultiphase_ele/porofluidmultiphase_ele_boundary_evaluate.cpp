/*----------------------------------------------------------------------*/
/*!
 \file porofluidmultiphase_ele_boundary_evaluate.cpp

 \brief evaluation methods of the porofluidmultiphase boundary element

   \level 3

   \maintainer  Anh-Tu Vuong
                vuong@lnm.mw.tum.de
                http://www.lnm.mw.tum.de
                089 - 289-15251
 *----------------------------------------------------------------------*/

#include "porofluidmultiphase_ele.H"

#include "porofluidmultiphase_ele_action.H"
#include "porofluidmultiphase_ele_boundary_factory.H"
#include "porofluidmultiphase_ele_boundary_interface.H"

#include "../drt_lib/drt_element.H"
#include "../drt_lib/drt_discret.H"

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                             vuong 08/16 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::PoroFluidMultiPhaseBoundary::Evaluate(
    Teuchos::ParameterList&   params,
    DRT::Discretization&      discretization,
    std::vector<int>&         lm,
    Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2,
    Epetra_SerialDenseVector& elevec3)
{
  dserror("not implemented. Use the Evaluate() method with Location Array instead!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                             vuong 08/16 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::PoroFluidMultiPhaseBoundary::Evaluate(
    Teuchos::ParameterList&   params,
    DRT::Discretization&      discretization,
    LocationArray&            la,
    Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2,
    Epetra_SerialDenseVector& elevec3)
{
  // we assume here, that numdofpernode is equal for every node within
  // the element and does not change during the computations
  const int numdofpernode = NumDofPerNode(*(Nodes()[0]));

  // all physics-related stuff is included in the implementation class that can
  // be used in principle inside any element (at the moment: only Transport
  // boundary element)
  // If this element has special features/ methods that do not fit in the
  // generalized implementation class, you have to do a switch here in order to
  // call element-specific routines
  return DRT::ELEMENTS::PoroFluidMultiPhaseBoundaryFactory::
      ProvideImpl(this,numdofpernode,discretization.Name())->Evaluate(
      this,
      params,
      discretization,
      la,
      elemat1,
      elemat2,
      elevec1,
      elevec2,
      elevec3
      );
}


/*----------------------------------------------------------------------*
 | evaluate Neumann boundary condition on boundary element   vuong 08/16 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::PoroFluidMultiPhaseBoundary::EvaluateNeumann(
    Teuchos::ParameterList&   params,
    DRT::Discretization&      discretization,
    DRT::Condition&           condition,
    std::vector<int>&         lm,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseMatrix* elemat1)
{
  // add Neumann boundary condition to parameter list
  params.set<DRT::Condition*>("condition",&condition);

  // build the location array
  LocationArray la(discretization.NumDofSets());
  DRT::Element::LocationVector(discretization,la,false);

  // evaluate boundary element
  return Evaluate(params,discretization,la,*elemat1,*elemat1,elevec1,elevec1,elevec1);
}


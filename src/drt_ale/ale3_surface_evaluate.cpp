/*----------------------------------------------------------------------------*/
/*! \file

\brief Evaluate 3D ALE element

\level 1

*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#include "ale3.H"

#include "drt_discret.H"
#include "drt_utils.H"

#include "position_array.H"

#include "drt_utils_fem_shapefunctions.H"
#include "drt_utils_boundary_integration.H"

#include "inpar_parameterlist_utils.H"

#include "drt_element_integration_select.H"

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
DRT::ELEMENTS::Ale3Surface_Impl_Interface* DRT::ELEMENTS::Ale3Surface_Impl_Interface::Impl(
    DRT::ELEMENTS::Ale3Surface* ele)
{
  switch (ele->Shape())
  {
    case DRT::Element::quad4:
    {
      return DRT::ELEMENTS::Ale3Surface_Impl<DRT::Element::quad4>::Instance(
          ::UTILS::SingletonAction::create);
    }
    default:
      dserror("shape %d (%d nodes) not supported", ele->Shape(), ele->NumNode());
      break;
  }
  return NULL;
}

template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::Ale3Surface_Impl<distype>* DRT::ELEMENTS::Ale3Surface_Impl<distype>::Instance(
    ::UTILS::SingletonAction action)
{
  static auto singleton_owner = ::UTILS::MakeSingletonOwner(
      []()
      {
        return std::unique_ptr<DRT::ELEMENTS::Ale3Surface_Impl<distype>>(
            new DRT::ELEMENTS::Ale3Surface_Impl<distype>());
      });

  return singleton_owner.Instance(action);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int DRT::ELEMENTS::Ale3Surface::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm, Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2, Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2, Epetra_SerialDenseVector& elevec3)
{
  const Ale3::ActionType act = DRT::INPUT::get<Ale3::ActionType>(params, "action");

  switch (act)
  {
    case Ale3::ba_calc_ale_node_normal:
    {
      Teuchos::RCP<const Epetra_Vector> dispnp;
      std::vector<double> mydispnp;

      dispnp = discretization.GetState("dispnp");

      if (dispnp != Teuchos::null)
      {
        mydispnp.resize(lm.size());
        DRT::UTILS::ExtractMyValues(*dispnp, mydispnp, lm);
      }

      Ale3Surface_Impl_Interface::Impl(this)->ElementNodeNormal(
          this, params, discretization, lm, elevec1, mydispnp);

      break;
    }
    default:
      dserror("Unknown type of action '%i' for Ale3Surface", act);
      break;
  }  // end of switch(act)

  return 0;
}  // end of DRT::ELEMENTS::Ale3Surface::Evaluate

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int DRT::ELEMENTS::Ale3Surface::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    Epetra_SerialDenseVector& elevec1, Epetra_SerialDenseMatrix* elemat1)
{
  return 0;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
inline void DRT::ELEMENTS::Ale3Surface_Impl<distype>::ElementNodeNormal(Ale3Surface* ele,
    Teuchos::ParameterList& params, DRT::Discretization& discretization, std::vector<int>& lm,
    Epetra_SerialDenseVector& elevec1, std::vector<double>& mydispnp)
{
  DRT::UTILS::ElementNodeNormal<distype>(funct_, deriv_, fac_, unitnormal_, drs_, xsi_, xyze_, ele,
      discretization, elevec1, mydispnp, false, true);
}
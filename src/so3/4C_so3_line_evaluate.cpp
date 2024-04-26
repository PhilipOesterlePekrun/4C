/*----------------------------------------------------------------------*/
/*! \file

\brief line element

\level 1

*----------------------------------------------------------------------*/

#include "4C_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_lib_discret.hpp"
#include "4C_lib_elements_paramsinterface.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_so3_line.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------*
 * Integrate a Line Neumann boundary condition (public)         gee 04/08|
 * ----------------------------------------------------------------------*/
int DRT::ELEMENTS::StructuralLine::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseMatrix* elemat1)
{
  // set the interface ptr in the parent element
  ParentElement()->SetParamsInterfacePtr(params);
  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live
  };
  LoadType ltype;

  // spatial or material configuration depends on the type of load
  // currently only material frame used
  // enum Configuration
  //{
  //  config_none,
  //  config_material,
  //  config_spatial,
  //  config_both
  //};
  // Configuration config = config_none;

  const auto* type = condition.Get<std::string>("type");
  if (*type == "neum_live")
  {
    ltype = neum_live;
    // config = config_material;
  }
  else
    FOUR_C_THROW("Unknown type of LineNeumann condition");

  // get values and switches from the condition
  const auto* onoff = condition.Get<std::vector<int>>("onoff");
  const auto* val = condition.Get<std::vector<double>>("val");
  const auto* spa_func = condition.Get<std::vector<int>>("funct");

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  double time = -1.0;
  if (ParentElement()->IsParamsInterface())
    time = ParentElement()->ParamsInterfacePtr()->GetTotalTime();
  else
    time = params.get("total time", -1.0);

  const int numdim = 3;

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff->size()) < numdim)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  for (int checkdof = numdim; checkdof < int(onoff->size()); ++checkdof)
  {
    if ((*onoff)[checkdof] != 0)
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evalutaion is 3. Further DoFs are not considered.");
  }

  // element geometry update - currently only material configuration
  const int numnode = NumNode();
  CORE::LINALG::SerialDenseMatrix x(numnode, numdim);
  MaterialConfiguration(x);

  // integration parameters
  const CORE::FE::IntegrationPoints1D intpoints(gaussrule_);
  CORE::LINALG::SerialDenseVector shapefcts(numnode);
  CORE::LINALG::SerialDenseMatrix deriv(1, numnode);
  const CORE::FE::CellType shape = Shape();

  // integration
  for (int gp = 0; gp < intpoints.nquad; ++gp)
  {
    // get shape functions and derivatives of element surface
    const double e = intpoints.qxg[gp][0];
    CORE::FE::shape_function_1D(shapefcts, e, shape);
    CORE::FE::shape_function_1D_deriv1(deriv, e, shape);
    switch (ltype)
    {
      case neum_live:
      {  // uniform load on reference configuration

        double dL;
        LineIntegration(dL, x, deriv);

        // loop the dofs of a node
        for (int i = 0; i < numdim; ++i)
        {
          if ((*onoff)[i])  // is this dof activated?
          {
            // factor given by spatial function
            const int functnum = (spa_func) ? (*spa_func)[i] : -1;
            double functfac = 1.0;

            if (functnum > 0)
            {
              // calculate reference position of GP
              CORE::LINALG::SerialDenseMatrix gp_coord(1, numdim);
              CORE::LINALG::multiplyTN(gp_coord, shapefcts, x);

              // write coordinates in another datatype
              double gp_coord2[numdim];
              for (int k = 0; k < numdim; k++) gp_coord2[k] = gp_coord(0, k);
              const double* coordgpref = gp_coord2;  // needed for function evaluation

              // evaluate function at current gauss point
              functfac = GLOBAL::Problem::Instance()
                             ->FunctionById<CORE::UTILS::FunctionOfSpaceTime>(functnum - 1)
                             .Evaluate(coordgpref, time, i);
            }

            const double fac = (*val)[i] * intpoints.qwgt[gp] * dL * functfac;
            for (int node = 0; node < numnode; ++node)
            {
              elevec1[node * numdim + i] += shapefcts[node] * fac;
            }
          }
        }

        break;
      }

      default:
        FOUR_C_THROW("Unknown type of LineNeumann load");
        break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *  (private)                                                  gee 04/08|
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralLine::LineIntegration(double& dL,
    const CORE::LINALG::SerialDenseMatrix& x, const CORE::LINALG::SerialDenseMatrix& deriv)
{
  // compute dXYZ / drs
  CORE::LINALG::SerialDenseMatrix dxyzdrs(1, 3);
  CORE::LINALG::multiply(dxyzdrs, deriv, x);
  dL = 0.0;
  for (int i = 0; i < 3; ++i) dL += dxyzdrs(0, i) * dxyzdrs(0, i);
  dL = sqrt(dL);
  return;
}

FOUR_C_NAMESPACE_CLOSE
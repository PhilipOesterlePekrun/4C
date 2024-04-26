/*----------------------------------------------------------------------------*/
/*! \file

\brief three dimensional nonlinear torsionless rod based on a C1 curve

\level 2

*/
/*----------------------------------------------------------------------------*/

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_material.hpp"
#include "4C_mat_par_parameter.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Beam3eb::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  if (Material()->Parameter()->Name() != "MAT_BeamKirchhoffTorsionFreeElastHyper" and
      Material()->Parameter()->Name() != "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes")
  {
    FOUR_C_THROW(
        "The material parameter definition '%s' is not supported by "
        "Beam3eb element! "
        "Choose MAT_BeamKirchhoffTorsionFreeElastHyper or "
        "MAT_BeamKirchhoffTorsionFreeElastHyper_ByModes!",
        Material()->Parameter()->Name().c_str());
  }

  return true;
}

FOUR_C_NAMESPACE_CLOSE
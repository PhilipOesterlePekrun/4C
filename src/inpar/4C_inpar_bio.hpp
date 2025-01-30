// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_BIO_HPP
#define FOUR_C_INPAR_BIO_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}

// ToDo: move these enums to namespace Inpar::ArteryNetwork etc.
//       is the typedef really needed?

/*!----------------------------------------------------------------------
\brief enum of arterial network dynamic types

\author ismail \date 11/08
This is the enumeration of all types of different integration schemes

*-----------------------------------------------------------------------*/


/*!----------------------------------------------------------------------
\brief enum of reduced dimensional airways dynamic types

\author ismail \date 01/10
This is the enumeration of all types of different integration schemes

*-----------------------------------------------------------------------*/
typedef enum RedAirwaysDyntype
{
  one_step_theta,
  linear,
  nonlinear
} _RED_AIRWAYS_DYNTYPE;


namespace Inpar
{
  namespace ArtDyn
  {
    enum TimeIntegrationScheme
    {
      tay_gal,
      stationary
    };

    /// initial field for artery problem
    enum InitialField
    {
      initfield_zero_field,
      initfield_field_by_function,
      initfield_field_by_condition
    };

    //! element implementation type
    enum ImplType
    {
      impltype_undefined,
      impltype_lin_exp,
      impltype_pressure_based
    };

    /// set the arterial dynamic parameters
    void set_valid_parameters(Teuchos::ParameterList& list);
  }  // namespace ArtDyn

  namespace ArteryNetwork
  {
    /*!----------------------------------------------------------------------
    \brief enum of reduced dimensional relaxation type

    \author roth \date 10/13
    This is the enumeration of all types of different relaxation types

    *-----------------------------------------------------------------------*/
    enum Relaxtype3D0D
    {
      norelaxation,
      fixedrelaxation,
      Aitken,
      SD
    };

    //! type of coupling between artery network and poromultiphasescatra-framework
    enum ArteryPoroMultiphaseScatraCouplingMethod
    {
      none,   // none
      nodal,  // nodal
      gpts,   // Gauss-Point-To-Segment
      mp,     // Mortar-Penalty
      ntp     // 1Dnode-to-point in 2D/3D
    };

    /// set the artnet parameters
    void set_valid_parameters(Teuchos::ParameterList& list);

    /// set specific artnet conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  }  // namespace ArteryNetwork

  namespace BioFilm
  {
    /// set the biofilm parameters
    void set_valid_parameters(Teuchos::ParameterList& list);

    /// set specific biofilm conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
  }  // namespace BioFilm

  namespace ReducedLung
  {
    /// set the reduced airways parameters
    void set_valid_parameters(Teuchos::ParameterList& list);

    /// set specific reduced airways conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
  }  // namespace ReducedLung
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif

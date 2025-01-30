// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_STI_HPP
#define FOUR_C_INPAR_STI_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration

namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace Inpar
{
  namespace STI
  {
    //! type of coupling between scatra and thermo fields
    enum class CouplingType
    {
      undefined,
      monolithic,
      oneway_scatratothermo,
      oneway_thermotoscatra,
      twoway_scatratothermo,
      twoway_scatratothermo_aitken,
      twoway_scatratothermo_aitken_dofsplit,
      twoway_thermotoscatra,
      twoway_thermotoscatra_aitken
    };

    //! type of scalar transport time integration
    enum class ScaTraTimIntType
    {
      standard,
      elch
    };

    //! set valid parameters for scatra-thermo interaction
    void set_valid_parameters(Teuchos::ParameterList& list);

    //! set valid conditions for scatra-thermo interaction
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
  }  // namespace STI
}  // namespace Inpar
FOUR_C_NAMESPACE_CLOSE

#endif

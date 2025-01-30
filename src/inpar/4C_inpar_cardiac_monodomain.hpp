// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_CARDIAC_MONODOMAIN_HPP
#define FOUR_C_INPAR_CARDIAC_MONODOMAIN_HPP


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
  namespace ElectroPhysiology
  {
    /// possible types of evaluation of reaction term
    enum EvalType
    {
      ep_implicit,
      ep_semi_implicit,
    };

    /// set the elch parameters
    void set_valid_parameters(Teuchos::ParameterList& list);

    /// set specific elch conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
  }  // namespace ElectroPhysiology
}  // namespace Inpar
FOUR_C_NAMESPACE_CLOSE

#endif

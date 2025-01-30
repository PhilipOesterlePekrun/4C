// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_ELEMAG_HPP
#define FOUR_C_INPAR_ELEMAG_HPP

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
  namespace EleMag
  {
    /// Type of time integrator
    enum DynamicType
    {
      /// one-step-theta time integration
      elemag_ost,
      /// explicit euler method
      elemag_explicit_euler,
      /// implicit euler method
      elemag_bdf1,
      /// BDF2
      elemag_bdf2,
      /// BDF4
      elemag_bdf4,
      /// Generalized-Alpha method
      elemag_genAlpha,
      /// runge-kutta method
      elemag_rk,
      /// crank-nicolson method
      elemag_cn
    };

    /// Initial field for electromagnetic problems.
    enum InitialField
    {
      /// Initialize a zero field on all the components
      initfield_zero_field,
      /// Initialize the components as specified by the function
      initfield_field_by_function,
      /// Initialize the electric field with a CG scatra solution
      initfield_scatra,
      /// Initialize the electric field with a HDG scatra solution
      initfield_scatra_hdg
    };

    /// Define all valid parameters for electromagnetic problem.
    void set_valid_parameters(Teuchos::ParameterList& list);

    /// Set specific electromagnetic conditions.
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  }  // namespace EleMag
}  // namespace Inpar


FOUR_C_NAMESPACE_CLOSE

#endif

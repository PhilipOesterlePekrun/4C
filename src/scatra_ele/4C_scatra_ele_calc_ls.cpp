// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_ls.hpp"

#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcLS<distype>* Discret::Elements::ScaTraEleCalcLS<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcLS<distype>>(
            new ScaTraEleCalcLS<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcLS<distype>::ScaTraEleCalcLS(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname)
{
  // safety check
  if (my::scatrapara_->rb_sub_gr_vel())
    FOUR_C_THROW("CalcSubgrVelocityLevelSet not available anymore");

  return;
}


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::pyramid5>;
// template class Discret::Elements::ScaTraEleCalcLS<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

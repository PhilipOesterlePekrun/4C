// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// tpetra_fixed_hash_table_openmp_symbols.cpp

#include <Kokkos_Core.hpp>
#include <Tpetra_Details_FixedHashTable_decl.hpp>
#include <Tpetra_Details_FixedHashTable_def.hpp>

namespace Tpetra::Details
{
  template class FixedHashTable<int, int, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>;
}

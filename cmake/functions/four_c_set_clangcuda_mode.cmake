# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Kokkos uses the RULE_LAUNCH properties to redirect compilation through kokkos_launch_compiler.
# Clear only those properties. Generic CMake compiler launchers are user-controlled and remain
# enabled; integrations that must launch the final Clang command can instead use the dedicated
# CLANGCUDA_COMPILER_LAUNCHER environment variable.
# clangcuda_mode can be either CLANGCUDA_MODE_HOST or CLANGCUDA_MODE_DEVICE
function(set_clangcuda_mode target clangcuda_mode)
  set_target_properties(${target} PROPERTIES RULE_LAUNCH_COMPILE "" RULE_LAUNCH_LINK "")
  target_compile_definitions(${target} PRIVATE ${clangcuda_mode})
endfunction()

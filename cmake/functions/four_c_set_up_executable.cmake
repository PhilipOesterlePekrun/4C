function(four_c_disable_kokkos_launcher target_name)
  if(TARGET ${target_name})
    set_target_properties(${target_name} PROPERTIES
      CXX_COMPILER_LAUNCHER ""
      C_COMPILER_LAUNCHER ""
      CUDA_COMPILER_LAUNCHER ""
      RULE_LAUNCH_COMPILE ""
      RULE_LAUNCH_LINK ""
    )

    get_target_property(_cxx_launcher ${target_name} CXX_COMPILER_LAUNCHER)
    message(VERBOSE "${target_name}: CXX_COMPILER_LAUNCHER='${_cxx_launcher}'")
  endif()
endfunction()

# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Call this function on an executable of this project.
# The executable will link to the main library and use the internal compiler settings.
function(four_c_set_up_executable target)
  target_link_libraries(${target} PRIVATE ${FOUR_C_LIBRARY_NAME})
  target_link_libraries(${target} PRIVATE four_c_private_compile_interface)
  
  four_c_disable_kokkos_launcher(${target})

  target_compile_definitions(${target} PRIVATE
    FOUR_C_CLANGCUDA_HOST_ONLY
  )
  message(STATUS "-=- ${target}: Clang CUDA host-only compile")
  
endfunction()

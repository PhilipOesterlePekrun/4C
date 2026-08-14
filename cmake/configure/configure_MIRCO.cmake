# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_process_global_option(
  FOUR_C_MIRCO_FIND_INSTALLED
  DESCRIPTION
  "Use installed MIRCO instead of fetching sources"
  DEFAULT
  OFF
  )
if(FOUR_C_MIRCO_FIND_INSTALLED)
  # Note that MIRCO and 4C must point to the same Kokkos and Kokkos-Kernels installation. Otherwise, there will be errors.
  message(STATUS "FOUR_C_MIRCO_FIND_INSTALLED is enabled")

  # MIRCO provides a package configuration file if installed.
  find_package(mirco_lib HINTS ${FOUR_C_MIRCO_ROOT})

  if(NOT mirco_lib_FOUND)
    message(
      FATAL_ERROR
        "mirco_lib could not be found. Please ensure that the FOUR_C_MIRCO_ROOT path is correctly defined in the config file. Also, please use 'make install' and not just 'make' to install MIRCO."
      )
  endif()

else() # Fetch MIRCO from GIT repository
  # Turn off googletest in MIRCO so that it does not interfere with 4C.
  set(GTEST_IN_MIRCO "OFF")
  # Explicitly turn off `*_IN_MIRCO`, so that MIRCO uses upstream targets
  set(RYML_IN_MIRCO "OFF")
  set(KOKKOS_IN_MIRCO "OFF")
  set(KOKKOS_KERNELS_IN_MIRCO "OFF")

  # Propagate
  if(FOUR_C_CLANGCUDA)
    set(MIRCO_CLANGCUDA "ON")
  else()
    set(MIRCO_CLANGCUDA "OFF")
  endif()

  set(MIRCO_GIT_REPO "https://github.com/imcs-compsim/MIRCO.git")
  set(MIRCO_GIT_TAG "8b049a6462eba5809d7cffe039a77f3bc5593767") # latest hash 02.06.2026

  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
  fetchcontent_declare(
    mirco
    GIT_REPOSITORY ${MIRCO_GIT_REPO}
    GIT_TAG ${MIRCO_GIT_TAG}
    )

  # MIRCO's Clang-CUDA configuration clears compiler and rule launchers in its parent scope.
  # Preserve 4C's launcher configuration while MIRCO configures its own targets.
  if(FOUR_C_CLANGCUDA)
    set(_four_c_cxx_compiler_launcher "${CMAKE_CXX_COMPILER_LAUNCHER}")
    get_property(
      _four_c_rule_launch_compile_is_set GLOBAL
      PROPERTY RULE_LAUNCH_COMPILE
      SET
      )
    get_property(_four_c_rule_launch_compile GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
    get_property(
      _four_c_rule_launch_link_is_set GLOBAL
      PROPERTY RULE_LAUNCH_LINK
      SET
      )
    get_property(_four_c_rule_launch_link GLOBAL PROPERTY RULE_LAUNCH_LINK)
  endif()

  fetchcontent_makeavailable(mirco)

  if(FOUR_C_CLANGCUDA)
    set(CMAKE_CXX_COMPILER_LAUNCHER
        "${_four_c_cxx_compiler_launcher}"
        CACHE STRING "C++ compiler launcher" FORCE
        )

    if(_four_c_rule_launch_compile_is_set)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${_four_c_rule_launch_compile}")
    else()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
    endif()
    if(_four_c_rule_launch_link_is_set)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${_four_c_rule_launch_link}")
    else()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK)
    endif()
  endif()

  # MIRCO requires a specific path, possibly due to inconsistent naming "mirco" vs "mirco_lib".
  set(FOUR_C_MIRCO_ROOT "${CMAKE_INSTALL_PREFIX}/lib/cmake/mirco")
endif()

four_c_add_external_dependency(four_c_all_enabled_external_dependencies mirco::mirco_lib)

four_c_remember_variable_for_install(FOUR_C_MIRCO_ROOT)

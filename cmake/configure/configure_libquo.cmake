# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

set(libquo_ROOT
    "/home/oesterle/rd/tpl/libquo_base/install"
    CACHE PATH "Path to libquo installation"
    )

find_path(libquo_INCLUDE_DIR
  NAMES quo.h
  PATHS "${libquo_ROOT}/include"
  NO_DEFAULT_PATH
  REQUIRED
  )

find_library(libquo_LIBRARY
  NAMES quo
  PATHS "${libquo_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED
  )

message(STATUS "libquo_ROOT: ${libquo_ROOT}")
message(STATUS "libquo_INCLUDE_DIR: ${libquo_INCLUDE_DIR}")
message(STATUS "libquo_LIBRARY: ${libquo_LIBRARY}")

add_library(libquo::quo UNKNOWN IMPORTED GLOBAL)

set_target_properties(libquo::quo PROPERTIES
  IMPORTED_LOCATION "${libquo_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${libquo_INCLUDE_DIR}"
  INTERFACE_LINK_OPTIONS "-Wl,-rpath,${libquo_ROOT}/lib"
  )

set(FOUR_C_WITH_LIBQUO
    TRUE
    CACHE BOOL "Whether 4C was built with libquo support" FORCE
    )

set(libquo_additional_configuration
    FOUR_C_WITH_LIBQUO
    )

target_link_libraries(
  four_c_all_enabled_external_dependencies INTERFACE libquo::quo
  )

four_c_remember_variable_for_install(
  libquo_ROOT
  libquo_INCLUDE_DIR
  libquo_LIBRARY
  )

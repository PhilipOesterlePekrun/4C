# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

if(FOUR_C_CLANGCUDA)
  four_c_auto_define_module(CLANGCUDA_MODE CUDA_DEVICE)
  message(STATUS "//###### four_c_auto_define_module(CLANGCUDA_MODE CUDA_DEVICE) contact")
else()
  four_c_auto_define_module()
endif()


set(_dependencies
    # cmake-format: sortable
    config
    contact_constitutivelaw
    core
    global_data
    inpar
    mat
    mortar
    scatra
    scatra_ele
    solver_nonlin_nox
    structure_new
    xfem
    )

four_c_add_internal_dependency(${AUTO_DEFINED_MODULE_NAME} ${_dependencies})

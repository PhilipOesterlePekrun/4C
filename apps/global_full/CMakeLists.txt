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

set(OBJS_FOUR_C_MAIN
    # cmake-format: sortable
    4C_global_full_entrypoint_switch.cpp
    4C_global_full_io.cpp
    4C_global_full_main.cpp
    )

add_executable(${FOUR_C_EXECUTABLE_NAME} ${OBJS_FOUR_C_MAIN})

four_c_disable_kokkos_launcher(${FOUR_C_EXECUTABLE_NAME})
target_compile_definitions(${FOUR_C_EXECUTABLE_NAME} PRIVATE
  FOUR_C_CLANGCUDA_HOST_ONLY
)

set_target_properties(
  ${FOUR_C_EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  )
four_c_set_up_executable(${FOUR_C_EXECUTABLE_NAME})

four_c_disable_kokkos_launcher(${FOUR_C_EXECUTABLE_NAME})
target_compile_definitions(${FOUR_C_EXECUTABLE_NAME} PRIVATE
  FOUR_C_CLANGCUDA_HOST_ONLY
)

if(FOUR_C_ENABLE_METADATA_GENERATION)
  if(FOUR_C_WITH_PYTHON)
    add_custom_command(
      TARGET ${FOUR_C_EXECUTABLE_NAME}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E rm -f ${PROJECT_BINARY_DIR}/4C_metadata.yaml
        ${PROJECT_BINARY_DIR}/4C_schema.json ${PROJECT_BINARY_DIR}/4C_schema_partial.json
      COMMAND
        ${FOUR_C_ENABLE_ADDRESS_SANITIZER_TEST_OPTIONS} "$<TARGET_FILE:${FOUR_C_EXECUTABLE_NAME}>"
        -p > ${PROJECT_BINARY_DIR}/4C_metadata.yaml
      COMMAND
        ${FOUR_C_PYTHON_VENV_BUILD}/bin/create-schema-files ${PROJECT_BINARY_DIR}/4C_metadata.yaml
        ${PROJECT_BINARY_DIR}/4C_schema.json
      COMMENT "Creating metadata and JSON schema files"
      )
  else()
    message(
      FATAL_ERROR
        "Specifying FOUR_C_ENABLE_METADATA_GENERATION requires FOUR_C_WITH_PYTHON to be enabled."
      )
  endif()
endif()

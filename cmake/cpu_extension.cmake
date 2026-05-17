function(add_vkwr_cpu_extension ext_name)
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    add_library(${ext_name} MODULE ${ARG_SOURCES})

    target_link_libraries(${ext_name} PRIVATE ${TORCH_LIBRARIES})

    set_target_properties(${ext_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/vkwr"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

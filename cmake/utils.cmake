function(add_vkwr_extension ext_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINK_LIBRARIES;DEFINITIONS" ${ARGN})

    set(SOURCES ${ARG_SOURCES})
    set(DEFINITIONS ${ARG_DEFINITIONS})
    set(LINK_LIBS ${ARG_LINK_LIBRARIES})

    Python_add_library(${ext_name} MODULE WITH_SOABI ${SOURCES})

    target_include_directories(${ext_name} PRIVATE "${CMAKE_SOURCE_DIR}/csrc")
    target_link_libraries(${ext_name} PRIVATE torch ${LINK_LIBS})

    target_compile_definitions(${ext_name} PRIVATE
        TORCH_EXTENSION_NAME=${ext_name})

    if(DEFINITIONS)
        target_compile_definitions(${ext_name} PRIVATE ${DEFINITIONS})
    endif()

    if(VKWR_TARGET_DEVICE STREQUAL "cuda")
        set_target_properties(${ext_name} PROPERTIES
            CUDA_ARCHITECTURES "${VKWR_CUDA_ARCH}"
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_options(${ext_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 --extra-device-vectorization -Xptxas -O3>
        )
    endif()

    set_target_properties(${ext_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/vkwr"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endfunction()

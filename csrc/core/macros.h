#ifndef VKWR_MACROS_H
#define VKWR_MACROS_H

#include <torch/library.h>

#define VKWR_LIBRARY(lib_name) \
    TORCH_LIBRARY(vkwr_##lib_name, m)

#define VKWR_LIBRARY_IMPL(lib_name) \
    TORCH_LIBRARY_IMPL(vkwr_##lib_name, BackendSelect, m) \
    TORCH_LIBRARY_IMPL(vkwr_##lib_name, CUDA, m)

#define VKWR_DEFINE(op_schema) m.def(#op_schema, op_schema)

#endif // VKWR_MACROS_H

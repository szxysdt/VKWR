#pragma once

#include <Python.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// Expand macro arguments (allows NAME to be a macro rather than a literal)
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// Lets .so be loadable by Python import (replaces empty PYBIND11_MODULE)
#define REGISTER_EXTENSION(NAME)                                              \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                    \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                          \
  }

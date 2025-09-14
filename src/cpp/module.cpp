// src/cpp/module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <torch/extension.h>  // for torch::Tensor in bindings

namespace py = pybind11;

PYBIND11_MODULE(hpdex_cpp, m) {
    m.doc() = "HPDEx C++ extension module";
}

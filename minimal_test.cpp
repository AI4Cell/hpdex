#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(minimal_test, m) {
    m.doc() = "pybind11 minimal test";
    m.def("add", &add, "A function which adds two numbers");
}

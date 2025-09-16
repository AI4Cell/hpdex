#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::tuple<int, int> test_function(int a, int b) {
    return std::make_tuple(a + b, a - b);
}

PYBIND11_MODULE(simple_module, m) {
    m.doc() = "Simple test module";
    m.def("test_function", &test_function, "A simple test function");
}

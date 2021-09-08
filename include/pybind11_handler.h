#pragma once

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>

using namespace pybind11::literals; // to bring in the `_a` literal

namespace py = pybind11;

namespace pybind_handler {
    py::object import(const std::string &module, const std::string &path, py::object &globals);
    bool is_callable(py::handle obj);
    std::string to_std_string(py::handle obj);
    py::object get_method(py::handle instance, const std::string &method_name);
}


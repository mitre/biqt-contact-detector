#include "pybind11_handler.h"


py::object pybind_handler::import(const std::string &module, const std::string &path, py::object &globals) {
    py::dict locals;
    locals["module_name"] = py::cast(module);
    locals["path"] = py::cast(path);

    py::eval<py::eval_statements>(
            "import importlib.util\n"
            "import importlib.machinery\n"
            "loader = importlib.machinery.SourceFileLoader(module_name, path)\n"
            "spec = importlib.util.spec_from_loader(loader.name, loader)\n"
            "new_module = importlib.util.module_from_spec(spec)\n"
            "loader.exec_module(new_module)\n",
            globals,
            locals);

    return locals["new_module"];
}

bool pybind_handler::is_callable(py::handle obj) {
    // This returns true for any callable, not just functions.
    return py::isinstance<py::function>(obj);
}

std::string pybind_handler::to_std_string(py::handle obj) {
    return py::str(obj);
}

py::object pybind_handler::get_method(py::handle instance, const std::string &method_name) {
    py::object method = py::getattr(instance, method_name.c_str(), py::none());
    if (method.is_none() || is_callable(method)) {
        return method;
    }else{
        throw std::invalid_argument("Method " + method_name + " provided is not callable.");
    }
}

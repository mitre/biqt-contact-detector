#include "ContactDetector.h"
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <dlfcn.h>
#include <fstream>

ContactDetector::ContactDetector(const std::string &module_file, 
                                 const std::string &module_name, 
                                 const std::string &module_object,
                                 const std::string &eval_method,
                                 const std::string &cosmetic_model_path,
                                 const std::string &soft_lens_model_path)
{
    // Load python library so numpy can find symbols. See
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    void *libpython_handle = dlopen("libpython3.10.so", RTLD_GLOBAL | RTLD_LAZY);
    if(!libpython_handle){
        throw std::runtime_error("Error loading python library libpython3.10.so");
    }

    if(!Py_IsInitialized()){
        py::initialize_interpreter();
    }

    py::object main = py::module::import("__main__");
    py::object globals = main.attr("__dict__");

    py::object module = pybind_handler::import(module_name, module_file, globals); 

    py::object ModelObj = module.attr(module_object.c_str());
    if (!pybind_handler::is_callable(ModelObj)) {
        throw std::invalid_argument("Object " + module_object + " from module " + module_name + " is not callable.");
    }
    
    py::object model_obj;
    if(soft_lens_model_path.empty()){
        model_obj = ModelObj(cosmetic_model_path);
        m_is_dual = false;
    }else{
        model_obj = ModelObj(cosmetic_model_path, soft_lens_model_path);
        m_is_dual = true;
    }
    m_eval_func = pybind_handler::get_method(model_obj, eval_method.c_str());

}

ContactDetector::~ContactDetector()
{

}

bool ContactDetector::is_dual(){
    return m_is_dual;
}


std::vector<double> ContactDetector::evaluate(const std::string &file){
    try{
        return m_eval_func(file).cast<std::vector<double>>();
    }catch(py::error_already_set &e){
        if(e.matches(PyExc_FileNotFoundError)){
            throw std::runtime_error("ERROR: Image " + file + " could not be found.");
        }else if(e.matches(PyExc_IOError)){
            throw std::runtime_error("ERROR: File " + file + " could not be read as an image.");
        }
    }
}

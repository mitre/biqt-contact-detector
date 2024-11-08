#ifndef CONTACTDETECTOR_H
#define CONTACTDETECTOR_H

#include <string>
#include "pybind11_handler.h"

class ContactDetector {

public:
    ContactDetector(const std::string &module_file, 
                    const std::string &module_name, 
                    const std::string &module_object,
                    const std::string &eval_method,
                    const std::string &cosmetic_model_path);

    ~ContactDetector();
    double evaluate(const std::string &file);


private:
    py::function m_eval_func;

};

#endif




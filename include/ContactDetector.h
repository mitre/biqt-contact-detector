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
                    const std::string &cosmetic_model_path,
                    const std::string &soft_lens_model_path=std::string());

    ~ContactDetector();
    std::vector<double> evaluate(const std::string &file);
    bool is_dual();


private:
    py::function m_eval_func;
    bool m_is_dual;

};

#endif




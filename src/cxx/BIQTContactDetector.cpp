// #######################################################################
// NOTICE
//
// This software (or technical data) was produced for the U.S. Government
// under contract, and is subject to the Rights in Data-General Clause
// 52.227-14, Alt. IV (DEC 2007).
//
// Copyright 2019 The MITRE Corporation. All Rights Reserved.
// #######################################################################

#include <BIQTContactDetector.h>
#include <fstream>
#include <map>

bool file_openable(std::string filename){
    return static_cast<bool>(std::ifstream(filename));
}

BIQTContactDetector::BIQTContactDetector()
{
    // Initialize metadata
    std::string biqt_home = getenv("BIQT_HOME");
    std::ifstream desc_file(biqt_home +
                                "/providers/BIQTContactDetector/descriptor.json",
                            std::ifstream::binary);
    desc_file >> DescriptorObject;

    // Instantiate model
    std::string cosmetic_model_path = biqt_home + "/providers/BIQTContactDetector/config/models/efficientnet-v2l-CDM.ckpt";
    
    if(!file_openable(cosmetic_model_path)){
        throw std::runtime_error("\nERROR: Cosmetic model file '" + cosmetic_model_path + "' could not be opened (does it exist?). Cosmetic model is needed to use this component.");
    }
    
    try{
        std::string contactClassifierNetwork_module_file = biqt_home + "/providers/BIQTContactDetector/src/python/inference/pytorch_detector.py";
        std::string contactClassifierNetwork_module_name = "pytorch_detector";
        std::string contactClassifierNetwork_module_object = "IrisDetection";
        std::string contactClassifierNetwork_eval_method = "infer";
        m_contact_detector = std::unique_ptr<ContactDetector>(new ContactDetector(contactClassifierNetwork_module_file, 
                                                                                    contactClassifierNetwork_module_name, 
                                                                                    contactClassifierNetwork_module_object, 
                                                                                    contactClassifierNetwork_eval_method,
                                                                                    cosmetic_model_path));
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        throw e;
    }

}

BIQTContactDetector::~BIQTContactDetector()
{
}

Provider::EvaluationResult BIQTContactDetector::evaluate(const std::string &file)
{

    // Initialize some variables
    Provider::EvaluationResult evalResult;
    Provider::QualityResult qualityResult;

    evalResult.errorCode = 0;
    evalResult.provider = "BIQTContactDetector";

    try{
        double cosmetic_contact_confidence = m_contact_detector->evaluate(file);
        
        qualityResult.metrics["cosmetic_contact_confidence"] = cosmetic_contact_confidence;
        evalResult.qualityResult.push_back(std::move(qualityResult));
    }catch(std::exception &e){
        evalResult.message = e.what();
        evalResult.errorCode = 1;
        std::cerr << e.what() << std::endl;
    }
    
    return evalResult;
}

bool file_exists(const std::string &file){
    std::ifstream ifile;
    ifile.open(file);
    if(ifile) {
        return true;
    } else {
        return false;
    }
}

std::unique_ptr<BIQTContactDetector> p; 
DLL_EXPORT const char *provider_eval(const char *cFilePath)
{
    std::string filePath(cFilePath);
    Provider::EvaluationResult result;

    // Return early if the file doesn't even exist
    if(!file_exists(filePath)){
        result.message = "ERROR: File " + filePath + " could not be found!";
        result.errorCode = 1;
        std::cerr << result.message << std::endl;
        return Provider::serializeResult(result);
    }

    // Try lazy loading model
    if(!p){
        try{
            p = std::unique_ptr<BIQTContactDetector>(new BIQTContactDetector());
            result = p->evaluate(filePath);
        }catch(std::exception &e){
            result.message = e.what();
            result.errorCode = 1;
            std::cerr << e.what() << std::endl;
        }
    }else{
        result = p->evaluate(filePath);
    }
    return Provider::serializeResult(result);
}
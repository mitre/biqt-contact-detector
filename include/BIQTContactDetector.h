// #######################################################################
// NOTICE
//
// This software (or technical data) was produced for the U.S. Government
// under contract, and is subject to the Rights in Data-General Clause
// 52.227-14, Alt. IV (DEC 2007).
//
// Copyright 2019 The MITRE Corporation. All Rights Reserved.
// #######################################################################

#ifndef BIQTCONTACTDETECTOR_H
#define BIQTCONTACTDETECTOR_H

#include <ProviderInterface.h>
#include <fstream>
#include <json/json.h>
#include <json/value.h>

#include <memory>
#include "ContactDetector.h"

class BIQTContactDetector : public Provider {

  public:
    BIQTContactDetector();
	  ~BIQTContactDetector() override;
    Provider::EvaluationResult evaluate(const std::string &file) override;

  private:
    std::unique_ptr<ContactDetector> m_contact_detector;
};

#endif

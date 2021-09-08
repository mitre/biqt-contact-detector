> NOTICE
>
> This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 
>
> (C) 2021 The MITRE Corporation. All Rights Reserved.
>
> Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.

## Overview

This guide provides instructions for the following scenarios:
  1. Creating a new cosmetic or soft contact lens prediction model using custom data collections (i.e., extending the model).
  1. Updating an existing cosmetic or soft contact lens prediction model with supplemental data (i.e., re-training the model).  

It describes the following high-level steps:
  1. Preparing the training environment
  1. Creating a training metadata file from a collection of labeled images 
  1. Building a model from the training images and metadata using the provided scripts
  1. Installing the model 

### Prepare the Environment

#### Dependencies

Extending or re-training the detection models requires a Linux host with Python 3.7 and CUDA libraries (if appropriate) 
installed.

Install required dependencies listed in this directory's `requirements.txt` using `pip`.

```
pip install -r requirements.txt
```

#### Quick Start with Docker

Use the provided Dockerfile and reference docker-compose file to quickly create a compatible environment for training.
  1. Rename the docker-compose.yml.example file to docker-compose.yml.
  1. Add one or more volumes to the docker-compose.yml file corresponding to the directories that contain the training data.
  1. Build the docker image by running `docker-compose build`.
  1. Start a new environment by running `docker-compose run biqt_contactlens_detection_trainer`.
  
### Create Training Metadata from Labeled Images

The training scripts used to extend or re-train the cosmetic and soft lens models each require training and
validation CSV files consisting of a header row of `image_path,contact_lens_type` followed by `image,label` pairs. 
Generally, an 80/20 to 90/10 split of labeled data between the training and validation files is desirable.

NOTE: Consistent labels must be used when extending existing models! The labels for the soft lens model 
are `Yes|No`, and the labels for the cosmetic lens CSV are `Cosmetic|Non-Cosmetic`.

| Target Model | Reference Training File | Reference Validation File |
| ------------ | ----------------------- | ------------------------- |
| Soft Lens | [metadata/train_softlens_example.csv](metadata/train_softlens_example.csv) | [metadata/validate_softlens_example.csv](metadata/validate_softlens_example.csv) |
| Cosmetic  | [metadata/train_cosmetic_example.csv](metadata/train_cosmetic_example.csv) | [metadata/validate_cosmetic_example.csv](metadata/validate_cosmetic_example.csv) |

### Training

Use the [train_scripts/train_cosmetic.sh.example](train_scripts/train_cosmetic.sh.example) and 
[train_scripts/train_softlens.sh.example](train_scripts/train_softlens.sh.example) scripts to extend or re-train 
the models. 

If re-training the model, comment out the `PREV_CHECKPOINT` declaration. The new or extended model, including logs and
performance/accuracy information from the training and validation phases, will be stored in `RESULT_DIR` 
(defaults to `results_cosmetic` or `results_softlens` depending on the script).

### Using New Models

Copy the model produced by the training script to the appropriate directory.

```bash
cp results_cosmetic/binary-cosmetic-contact-lens-model.hdf5 $BIQT_HOME/providers/BIQTContactDetector/config/models/
```  

... or ...

```bash
cp results_softlens/binary-clear-soft-contact-lens-model.hdf5 $BIQT_HOME/providers/BIQTContactDetector/config/models/
```

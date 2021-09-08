> NOTICE
> 
> This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007)
> 
> (C) 2021 The MITRE Corporation. All Rights Reserved.
> Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.

## Summary ##

BIQTContactDetector is a reference library for detecting contact lenses in individual iris images. It is part of the 
open source [Biometric Image Quality Toolkit Framework](https://github.com/mitre/biqt).

### Quality Attributes ###

The provider exposes the following attributes described in [descriptor.json](descriptor.json).

  * `cosmetic_contact_confidence` - A score in the range `[0,1]` indicating the predicted absence or presence (respectively) of a cosmetic lens in the image.
  * `soft_lens_confidence` (disabled by default) - A score in the range `[0,1]` indicating the predicted absence or presence (respectively) of a soft contact lens in the image.

### Models ###

The original models were trained on a combination of the Notre Dame [ND Cosmetic Contact Lenses 2013 Data Set (NDCLD13)](https://cvrl.nd.edu/projects/data/#nd-cosmetic-contact-lenses-2013-data-set) and [The Notre Dame Contact Lense Dataset 2015 (NDCLD15)](https://cvrl.nd.edu/projects/data/#the-notre-dame-contact-lense-dataset-2015ndcld15) 
datasets using an Nvidia V100 GPU. 

[This document](src/python/training/README.md) describes the process for training new models or enhancing existing ones (e.g., training
with additional data). 

### Soft Lens Model ###

The soft lens model is not included in this release and is disabled by default. If you want to enabled soft lens predictions, follow these steps:
1. Train a soft lens model following the instructions for training as described in [src/python/training/README.md] 
2. Copy the soft lens model hdf5 file to ```config/models/binary-clear-soft-contact-lens-model.hdf5```
3. When compiling via cmake, set the cache variable DUAL_NETWORK to "ON". E.g,: 
  ```
  cmake3 -DDUAL_NETWORK=ON ..
  ```
  There should be an indication in the cmake output that dual network (soft-lens + cosmetic-lens) model is enabled:
  ```
  -- Attempting to use local model files...
-- Detected local soft lens model /biqt-contact/config/models/binary-clear-soft-contact-lens-model.hdf5
-- Detected local cosmetic lens model /biqt-contact/config/models/binary-cosmetic-contact-lens-model.hdf5
-- Compiling with dual network enabled. <------
-- Configuring done
-- Generating done
-- Build files have been written to: /biqt-contact/build
  ```

The BIQTContactDetector should now output both cosmetic_contact_confidence and soft_lens_confidence:
```
[root@8a44d221c5dc biqt-contact]# biqt -p BIQTContactDetector data/example.tiff
Provider,Image,Detection,AttributeType,Key,Value
BIQTContactDetector,data/example.tiff,1,Metric,cosmetic_contact_confidence,6.98724e-07
BIQTContactDetector,data/example.tiff,1,Metric,soft_lens_confidence,1
```
  


### Acknowledgments ###

The execution of this component relies on the following dependencies. The software licenses for these dependencies are
provided in [LICENSE](LICENSE).  
  - [numpy](https://github.com/numpy/numpy)
  - [pillow](https://github.com/python-pillow/Pillow)
  - [pybind11](https://github.com/pybind/pybind11)
  - [scikit-image](https://github.com/scikit-image/scikit-image)
  - [scipy](https://github.com/scipy/scipy)
  - [tensorflow-probability](https://github.com/tensorflow/probability)
  - [tensorflow](https://github.com/tensorflow/tensorflow)

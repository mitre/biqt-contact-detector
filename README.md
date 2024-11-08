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

### Models ###

The model was trained on a combination of the Notre Dame [ND Cosmetic Contact Lenses 2013 Data Set (NDCLD13)](https://cvrl.nd.edu/projects/data/#nd-cosmetic-contact-lenses-2013-data-set) and [The Notre Dame Contact Lense Dataset 2015 (NDCLD15)](https://cvrl.nd.edu/projects/data/#the-notre-dame-contact-lense-dataset-2015ndcld15) datasets.

NOTE: The included cosmetic lens model was trained on a limited number of individuals with cosmetic lenses and should be treated only as an example. It should perform well on data similar to the Notre Dame contact lens datasets used for training, but it may not perform well on data captured with different image sensors. 

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

'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''

"""
Main file to run the contact lens detection tool

Sample Usage
python main.py -d "E:\Databases\Iris\Processed" -imgformat bmp
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
from PIL import UnidentifiedImageError

from pytorch_detector import IrisDetection

import glob
import datetime

def get_directory(base):
    return [x for x in glob.iglob(os.path.join(base, '*')) if os.path.isdir(x)]

def recursive_glob(base, pattern):
    """ Recursive walk starting in specified directory """
    imglist = []
    imglist.extend(glob.glob(os.path.join(base, pattern)))
    dirs = get_directory(base)
    if len(dirs):
        for d in dirs:
            imglist.extend(recursive_glob(os.path.join(base, d), pattern))
    return imglist

def get_img_files(path, ext):
    """Returns a list of all image files with extension ext"""
    return recursive_glob(path,'*.'+ext)

if __name__ == '__main__':

    filepath = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1], default=0, help='display verbosity level: 0 (default) and 1(high)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--image",default=None,help='Test image file path')
    group.add_argument("-d", "--dir",default=None,help='Directory containing test images')

    parser.add_argument("-imgformat","--imageformat",default='jpg', choices=['jpg','bmp','png','tiff'],help='File format used for searching images in the directory. Can be jpg, bmp, tiff, png. Default = jpg')
    parser.add_argument("-cosmetic_model_path", "--cosmetic_model_path",default=os.path.join(filepath,'models/trained-Simple-1.000.hdf5'),help='Path to cosmetic lens classifier keras model file. Default = code folder path')
    parser.add_argument("-output", "--output",default=os.path.join(filepath,'Result'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'.txt'),help=' Path for resultant output file where 0 means real image and 1 means attack image. Default = Result_{date}_{time}.txt The structure of the output file will be image name, 0/1')
    args = parser.parse_args()

    if args.verbosity == 1:
        print("Log Level - HIGH")

    if args.image is None and args.dir is None:
        parser.error("at least one of --image and --dir is required")
    elif args.dir is None:
        if os.path.exists(args.image) is False:
            parser.error("Provided image path does not exist")
        ListImages = [args.image]
        if args.verbosity == 1:
            print("Processing 1 image: ", args.image)

    elif args.image is None:
        if os.path.exists(args.dir) is False:
            parser.error("Provided directory path does not exist")
        ListImages = get_img_files(args.dir,args.imageformat)
        if args.verbosity == 1:
            print("Processing ", len(ListImages) , " images in : ", args.dir, ' of extension ', args.imageformat)

    if os.path.exists(args.cosmetic_model_path) is False:
        parser.error("Provided cosmetic model path does not exist" )

    if args.verbosity == 1:
        print("Using cosmetic classifier network")

    network = IrisDetection(args.cosmetic_model_path)
    all_preds = {}

    for image in ListImages:
        try:
            prediction = network.infer(image)
            if args.verbosity == 1:
                print("Inference on ", image, " = ", prediction)
            all_preds[image] = prediction
        except UnidentifiedImageError as e:
            print("Error: ", e, " in image: ", image)
    
    print(all_preds)

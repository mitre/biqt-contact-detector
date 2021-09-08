'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''

"""
Main file to run the contact lens detection tool

Sample Usage
python main.py -d "E:\Databases\Iris\Processed" -o Result.txt -imgformat bmp
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import numpy as np
import argparse
from ContactClassifierNetwork import ContactClassifierNetwork
from DualContactClassifierNetwork import DualContactClassifierNetwork
import glob
import datetime

def getdirectory(base):
    return [x for x in glob.iglob(os.path.join(base, '*')) if os.path.isdir(x)]

def recursiveglob(base, pattern):
    """ Recursive walk starting in specified directory """
    imglist = []
    imglist.extend(glob.glob(os.path.join(base, pattern)))
    dirs = getdirectory(base)
    if len(dirs):
        for d in dirs:
            imglist.extend(recursiveglob(os.path.join(base, d), pattern))
    return imglist

def getImgFiles(path, ext):
    """Returns a list of all image files with extension ext"""
    return recursiveglob(path,'*.'+ext)

if __name__ == '__main__':
    
    filepath = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1], default=0, help='display verbosity level: 0 (default) and 1(high)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--image",default=None,help='Test image file path')
    group.add_argument("-d", "--dir",default=None,help='Directory containing test images')
    
    parser.add_argument("-imgformat","--imageformat",default='jpg', choices=['jpg','bmp','png','tiff'],help='File format used for searching images in the directory. Can be jpg, bmp, tiff, png. Default = jpg')
    parser.add_argument("-cosmetic_model_path", "--cosmetic_model_path",default=os.path.join(filepath,'models/trained-Simple-1.000.hdf5'),help='Path to cosmetic lens classifier keras model file. Default = code folder path')
    parser.add_argument("-soft_lens_model_path", "--soft_lens_model_path",default=os.path.join(filepath,'models/trained-ResNet50_2-0.962.hdf5'),help='Path to soft lens classifier keras model file. Default = code folder path')
    parser.add_argument("-cosmetic_only", "--cosmetic_only", action="store_true", default=False,
                        help="Only perform cosmetic vs non-cosmetic lens classification. In this case, a label of '1' "
                             "means cosmetic and a label of '0' means non-cosmetic (could be either clear lens or non-lens)")
    parser.add_argument("-output", "--output",default=os.path.join(filepath,'Result'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'.txt'),help=' Path for resultant output file where 0 means real image and 1 means attack image. Default = Result_{date}_{time}.txt The structure of the output file will be image name, 0/1')
    args = parser.parse_args()
    
    if args.verbosity == 1:
        print("Log Level - HIGH")
        
    if args.image is None and args.dir is None:
        parser.error("at least one of --image and --dir is required")
    elif args.dir is None:
        if os.path.exists(args.image) == False:
            parser.error("Provided image path does not exist")        
        ListImages = [args.image]
        if args.verbosity == 1:
            print("Processing 1 image: ", args.image)
            
    elif args.image is None:
        if os.path.exists(args.dir) == False:
            parser.error("Provided directory path does not exist")       
        ListImages = getImgFiles(args.dir,args.imageformat)
        if args.verbosity == 1:
            print("Processing ", len(ListImages) , " images in : ", args.dir, ' of extension ', args.imageformat)
        
    if os.path.exists(args.cosmetic_model_path) == False:
        parser.error("Provided cosmetic model path does not exist" )

    if os.path.isfile(args.soft_lens_model_path) and not args.cosmetic_only:
        if args.verbosity == 1:
            print("Using dual classifier network")
        network = DualContactClassifierNetwork(args.cosmetic_model_path, args.soft_lens_model_path)
    else:
        if args.verbosity == 1:
            print("Using cosmetic classifier network")
        network = ContactClassifierNetwork(args.cosmetic_model_path)

    network.processFiles(ListImages, args.output , args.verbosity)
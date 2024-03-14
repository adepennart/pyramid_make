#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: pyramid_make.py
Date: March 7th, 2024
Author: Valentin Gillet and Auguste de Pennart
Description:
    makes image pyramids from image(s)

List of functions:
    file_sort
        sort numerically file list
    make_mask
        processing, more information online
    process_image
        makes image pyramid
        curerntly does not process image
    process_image_stack
        parralelizes process_image

List of "non standard modules"
    plot_functions:
        a module containing all functions to create 2d plot of neurons

Procedure:
    1. takes image folder as input
    2. makes image pyramid
    3. exports pyramid to output folder

Usage:
    pyramid_make.py [-h] [-v] -i INPUT_FOLDER -o OUTPUT_FOLDER
                         [-t TILE_SIZE] [-l LAYER_NUMBER]
                         [-d DOWNSCALE_FACTOR] [-c CORES]

known error:
    1. processing currently disabled

 """
import logging
import matplotlib.pyplot as plt
import numpy as np
import os, re, sys
import argparse


from multiprocessing import Pool
from PIL import Image
from skimage import data
from skimage.transform import pyramid_gaussian
from skimage import filters, exposure
from tqdm import tqdm
from scipy import ndimage

logging.basicConfig(level=logging.INFO)

Image.MAX_IMAGE_PIXELS = None

# func:sorts through files
# inputs:
#	file_list:
#		list of files to be sorted
#	sort_by_digit:
#		specified digit to sort by
#	rev:
#		places objects in descending order
# #outputs:
#	file_list:
#		list of sorted files/objects
def file_sort(file_list=None, sort_by_digit=0, rev=False):
	for n, filename in enumerate(file_list):
		for m, filename_2 in enumerate(file_list[n+1:len(file_list)]):
			try:
				match = int(re.findall("(\d+)", str(filename))[sort_by_digit])
				match_2 = int(re.findall(
					"(\d+)", str(filename_2))[sort_by_digit])
			except IndexError:
				print(" ERROR: Currently only works with filenames containing digits")
				sys.exit("Currently only works with filenames containing digits")
			if not rev:
				if match > match_2:
					temp_1 = filename
					temp_2 = filename_2
					filename = temp_2
					filename_2 = temp_1
					file_list[n] = temp_2
					file_list[n+m+1] = temp_1
			if rev:
				if n < n+1:
					temp_1 = filename
					temp_2 = filename_2
					filename = temp_2
					filename_2 = temp_1
					file_list[n] = temp_2
					file_list[n+m+1] = temp_1
	return file_list

def make_mask(image):
    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])
            
    mask_data = ndimage.binary_erosion(np.pad(image, 1),
                                       structure=kernel,
                                       iterations=1).astype(int)
    mask_data = ndimage.binary_fill_holes(mask_data)

    return mask_data[1:-1, 1:-1]


def process_image(file_path_dict):
    
    #splitting into two lists, for simplicity
    filepath=file_path_dict[0]
    meta_data=file_path_dict[1][0]
    file_num=file_path_dict[1][1]
    img  = Image.open(os.path.abspath(filepath))
    image = np.asarray(img)
    #pull up a try statement here for finding if windows or not
    z = filepath[:-4].split('/')[-1]
    if not z:
        z = filepath[:-4].split('\\')[-1]

    image_dir = os.path.join(meta_data[1], str(file_num))
    try:
        os.makedirs(image_dir, exist_ok=True)
    except:
        print('skipping')
        return False

    if meta_data[10]:
        mask = make_mask(image)
        # Gaussian + CLAHE, becomes float
        image = filters.gaussian(image, sigma=meta_data[4])
        image = exposure.equalize_adapthist(image, 
                                            kernel_size=meta_data[5], 
                                            nbins=meta_data[6], 
                                            clip_limit=meta_data[7])
        # Get back to 8bit
        image = (image*255).astype(np.uint8)
        image[np.logical_not(mask)] = 0
    
    # create pyramid array
    pyramid = tuple(pyramid_gaussian(image, 
                                     downscale=meta_data[9], 
                                     max_layer=meta_data[8]))  

    # For each pyramid layer, create tiles
    for i, layer in enumerate(pyramid):
        nested_dir=os.path.join(image_dir, str(i))
        os.makedirs(nested_dir, exist_ok=True)
        
        y_tiles, x_tiles = np.array(layer.shape) // meta_data[3]
        for y in range(y_tiles+1):
            for x in range(x_tiles+1):
                subarray = layer[y*meta_data[3]:(y+1)*meta_data[3], x*meta_data[3]:(x+1)*meta_data[3]]
                if not np.any(subarray > 0):
                    continue
                if subarray.shape != (meta_data[3], meta_data[3]):
                    # Pad array so it's correct tile size
                    diff = meta_data[3] - np.array(subarray.shape)
                    subarray = np.pad(subarray, 
                                      ((0, diff[0]), (0, diff[1])), 
                                      mode='constant', constant_values=0)

                subarray = (subarray*255).astype(np.uint8)  # Get back to 8bit
                # Save to file
                filename = os.path.join(nested_dir, f'{y}_{x}.jpg')
                im = Image.fromarray(subarray)
                im.save(filename, 'JPEG') 
    return True


def process_image_stack(files_dir,
                        n_workers,
                        out_dir,
                        process,
                        tile_size=512,
                        sigma_gauss=1,
                        kernel_CLAHE=[300,300],
                        nbins_CLAHE=256,
                        clip_CLAHE=1.7,
                        layer_limit_pyr=4,
                        downscaled_pyr=2):
    #making list of all viariables except files_dir, this will be added to all variables in inputs in the form of a dictionary, to allow iteration but also keeping the meta with each image
    files_dir = os.path.abspath(files_dir)
    files_list = file_sort(os.listdir(files_dir))
    main_dir = os.path.join(out_dir, 'pyramid')
    meta_data=[n_workers,main_dir,process,tile_size,sigma_gauss,kernel_CLAHE,nbins_CLAHE,clip_CLAHE,layer_limit_pyr,downscaled_pyr,process]
    os.makedirs(main_dir, exist_ok=True)

    inputs = [os.path.join(files_dir, f) for f in files_list]
    inputs_dict = {f:[meta_data,n] for n, f in enumerate(inputs)} # can we avoid making nested list?
    logging.info(f'Processing {len(inputs)} images from: {files_dir}')
    logging.info(f'Using {n_workers} workers')
    # Process images in parallel
    with Pool(n_workers) as p:
        results = list(tqdm(p.imap(process_image, inputs_dict.items()), total=len(inputs)))

    logging.info('Done!')
    logging.info(f'Output in: {main_dir}')
    return

if __name__ == '__main__':
    usage='make an image pyramd of inputted image(s)'
    parser=argparse.ArgumentParser(description=usage)#create an argument parser

    #creates the argument for program version
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 1.0')
    #creates the argument where input_folder will be inputted
    parser.add_argument('-i', '--input_folder',
                        metavar='INPUT_FOLDER',
                        dest='files_dir',
                        required=True,
                        help='input folder')
    #creates the argument where the output folder will be inputted
    parser.add_argument('-o', '--output_folder',
                        metavar='OUTPUT_FOLDER',
                        dest='out_dir',
                        required=True,
                        help='output folder')
    #creates the argument where the tile size will be inputted
    parser.add_argument('-t', '--tile_size',
                        metavar='TILE_SIZE',
                        dest='tile_size',
                        default=1024,
                        type=int,
                        help='the size of each individual tile making up an image')
    #creates the argument where the layer number be inputted
    parser.add_argument('-l', '--LAYER_number',
                        metavar='LAYER_NUMBER',
                        dest='layer_limit_pyr',
                        default=4,
                        type=int,
                        help='number of layers referring to depth of pyramid')
    #creates the argument where the downscale factor will be inputted
    parser.add_argument('-d', '--dowwnscale_factor',
                        metavar='DOWNSCALE_FACTOR',
                        dest='downscaled_pyr',
                        default=2,
                        type=int,
                        help='scaling factor between each layer')
    #creates the argument where the number of cores will be inputted
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='n_workers',
                        default=1,
                        type=int,
                        help='number of cores to use')
    args=parser.parse_args()#parses command line
    #currently does not process images
    process = False
    process_image_stack(os.path.abspath(args.files_dir),
                        args.n_workers,
                        os.path.abspath(args.out_dir),
                        process,
                        args.tile_size,
                        sigma_gauss=1,
                        kernel_CLAHE=[300,300],
                        nbins_CLAHE=256,
                        clip_CLAHE=1.7,
                        layer_limit_pyr=args.layer_limit_pyr,
                        downscaled_pyr=args.downscaled_pyr
                       )
    
    

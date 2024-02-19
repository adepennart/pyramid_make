"""
Title: pyramid_make.py

Date: Feb 19nd, 2024

Author: Auguste de Pennart

Description:
	makes a tiled image pyramid for one input image

List of functions:
    No user defined functions are used in the program.

List of "non standard modules"
    No user defined modules are used in the program.

Procedure:
    1. takes image as input and trnasforms into an array
    2. uses skimage to make array into pyramid object (array of several arrays)
    3. from defined tile size (currently set to 1024), tiles images of pyramid and exports them in the defined size

Usage:
	python pyramid_make.py
    or
    python3 pyramid_make.py

known error:
    1. numpy.ix might be faster than nested for loop
    2. does not take multiple images as input
    3. pad may be good to add
   	 
"""

#import packages
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_gaussian
from numpy import asarray

#variables
layer_limit=4 # how many layers to your layer pyramid
downscaled=2 # factor by which to downsample by
tile_size=1024 #size of tile to tile pyramid with
Image.MAX_IMAGE_PIXELS = None # disable default warning message
dir_name="output" #name of yout main dir

#import image(s)

try:  
    img  = Image.open("ov_z1_.tif")  #default input image
    # img  = Image.open("1.tif")  

    #try this later
    # img = np.array(Image.open("1.tif"), dtype=np.uint8)
except IOError: 
    pass

image = asarray(img) #turn image to array

#debug image show to make sure image is input as wanted
# fig, ax = plt.subplots()
# ax.imshow(image,cmap="gray")
# plt.show()

#array to pyramid array
pyramid = tuple(pyramid_gaussian(image, downscale=downscaled, max_layer=layer_limit))#, channel_axis=-1))
work_dir=os.getcwd()#get working current working directory for where to place output folder

#main dir is equivalent to dir 0 right now in trakem2
main_dir = os.path.join(work_dir, dir_name)  # make new directory
try:  # if error, directory already exists
    os.mkdir(main_dir)
except OSError:
    pass

for num in range(0,len(pyramid)): #goes through each layer of pyramid
    #this will be subnested
    nested_dir=os.path.join(main_dir,str(num))
    try:  # if error, directory already exists
        os.mkdir(nested_dir)
    except OSError:
        pass

    #make your image divisible by tile size by adding black counter to image
    rows, cols = pyramid[num].shape
    black_row=(((int(rows/tile_size)+1)*tile_size)-rows,cols)
    black_col=(rows+((int(rows/tile_size)+1)*tile_size)-rows,((int(cols/tile_size)+1)*tile_size)-cols)#check
    black_numpy_row=np.zeros(black_row,dtype=np.uint8)  
    black_numpy_col=np.zeros(black_col,dtype=np.uint8)
    added_pyramid=np.append(pyramid[num],black_numpy_row,axis = 0)
    added_pyramid=np.append(added_pyramid,black_numpy_col,axis = 1)
    rows, cols = added_pyramid.shape
    
    #go through each column and row and export image
    for n in range(0, int(rows/tile_size)):
        #define current tile of interest with max and min parameters
        row_max_tile=(n+1)*tile_size
        row_min_tile=(n+1)*tile_size-tile_size
        for col in range(0,int(cols/tile_size)):
            max_tile=(col+1)*tile_size
            min_tile=(col+1)*tile_size-tile_size
            image_arr = added_pyramid[row_min_tile:row_max_tile,min_tile:max_tile]*255
            im = Image.fromarray(image_arr) #make image
            im = im.convert("L")
            nested_file=os.path.join(nested_dir,str(n)+"_"+str(col)+".jpg") #make file path
            im.save(nested_file,"JPEG") #export
          

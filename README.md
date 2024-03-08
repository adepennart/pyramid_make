
## About
pipeline for making image pyramids.
This code is run on the terminal.

## Installation
This program can be directly installed from github (green Code button, top right).

Make sure to change into the downloaded directory, the code should resemble something like this.

```bash=
cd Downloads/pyramid_make
```

### Conda environment
First make sure conda is installed. If you do not have conda, refer to online resources on how to install conda.
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Once installed, we can make a conda environment.

```bash=
conda create --name pyramid_make
#activate
conda activate pyramid_make
```

### Python version
The python version for running this script is python= 3.12.1 
```bash=
conda install python= 3.12.1 
```

### Dependencies
The script runs with pip=24.0, pillow=10.2.0, numpy=1.26.3, matplotlib=3.8.2 and scikit-image=0.22.0.

Update your dependencies, if you do not already have the versions for these dependencies.

```bash=
pip install --upgrade pip==24.0

pip install pillow==10.2.0 numpy==1.26.3 matplotlib==3.8.2 scikit-image==0.22.0 tqdm==4.66.2
```

## Usage
### Input
The code can be run as follows. It has two required arguments; input and output folders.  
```bash=
    tif_to_pyramid.py [-h] [-v] -i INPUT_FOLDER -o OUTPUT_FOLDER
                         [-t TILE_SIZE] [-l LAYER_NUMBER]
                         [-d DOWNSCALE_FACTOR] [-c CORES]
```
More information about optional flags can be found with the following help command.
```bash=
    tif_to_pyramid.py -h
```

### example
The following command will use the example input and recreate the example output using two cores.
```bash=
python pyramid_make.py -i input/ -o output/ -c 2
```


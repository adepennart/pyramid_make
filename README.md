
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
The script runs with pip\==24.0, pillow\==10.2.0, numpy\==1.26.3, matplotlib\==3.8.2, scikit-image\==0.22.0 and python-catmaid==2.0.4

Update your dependencies, if you do not already have the versions for these dependencies.

```bash=
pip install --upgrade pip==21.3.1 wheel==0.37.1 setuptools==59.6.0

pip install python-catmaid==2.0.4 -U
```

## Usage
### Input
#change permissions in catmaid
The code can be run as follows. It uses default input provided (ov_z1_.tif) and outputs to output.
```bash=
   python pyramid_make.py
```


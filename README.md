# ParticleNet - LLP Fork

Implementation of the jet classification network in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570).

-----

## Getting started: ParticleNet on NAF GPU's

Instructions to use ParticleNet architecture with naf-gpu infrastructure at DESY.

**Note:** The full installation will take a while. Please be patient and carefully follow these instructions.

### Login

We will use the naf-gpu login node. Here `<username>` is your DESY username.

```bash
ssh -XY <username>@naf-cms-gpu01.desy.de
```

### Creating a link to your /nfs/dust/ area

This is needed to be able to see /nfs/dust/ files on JupyterHub.

```bash
cd ~
ln -s /nfs/dust/cms/user/<username> nfs_dust
```

### Installing anaconda

We will install all the needed packages with anaconda.

```bash
# load cuda module
module load cuda

# change directory to your home directory on dust
cd /nfs/dust/cms/user/<username>/

# get the anaconda installer
# Note: To have the latest version, you can look up the list at
# https://repo.continuum.io/archive/
wget https://repo.continuum.io/archive/Anaconda2-2019.10-Linux-x86_64.sh

# install anaconda
bash Anaconda2-2019.10-Linux-x86_64.sh

# press ENTER to review the license agreement and type "yes" to accept

# ATTENTION! When asked where to install anaconda,
# do NOT press enter to confirm the default location,
# but provide your dust home directory instead
# (type: /nfs/dust/cms/user/<username>/anaconda2).

# Answer all other prompts with the recommended option (in brackets).
# Optional: To have an easier way to activate your conda environment,
# you can allow anaconda to edit your .bashrc file.

# load anaconda
# IMPORTANT: You have to run this command every time you log in to NAF!
export PATH=/nfs/dust/cms/user/<username>/anaconda2/bin:$PATH

```


### Creating a conda environment

We will be working inside a conda environment. To create and activate the environment, follow the instructions below:

```bash
# create a conda environment called "particlenet"
conda create -n particlenet python=3.7

# activate the environment
# IMPORTANT: You also need to run this command every time you log in to NAF!
source activate /nfs/dust/cms/user/<username>/anaconda2/envs/particlenet
```

### Installing required packages

**Note:** When installing packages with conda, the "solving environment" step can take a long time. This is normal behavior so do not abort the installation (unless it runs longer than several hours).

```bash
# cd to your environment directory
# Note: This is important! If you try to install packages when not in
# your environment directory, you might get file I/O errors!
cd /nfs/dust/cms/user/<username>/anaconda2/envs/particlenet/

# install keras
conda install -c anaconda keras-gpu

# install tensorflow
conda install tensorflow

#install pandas (for data manipulation and analysis)
conda install pandas

# install matplotlib
conda install matplotlib

# install pytables
conda install pytables

# install scikit-learn
conda install scikit-learn

# install ROOT
conda install -c conda-forge root

# install root_numpy
conda install -c conda-forge root_numpy

# install awkward
conda install -c conda-forge awkward

# install uproot_methods
conda install -c conda-forge uproot-methods

# install jupyterhub kernel
cd /nfs/dust/cms/user/<username>/anaconda2/envs/particlenet #you should be here already, better to be sure
conda activate particlenet
pip install ipykernel --user
python -m ipykernel install --user --name="particlenet"
```

### Cloning this repository

If you followed these instructions, you should have tensorflow2.0 installed. Let's now clone this repository.

```bash
cd /nfs/dust/cms/user/<username>
mkdir ML_LLP #this is just my personal choice
cd ML_LLP
git clone https://github.com/lbenato/ParticleNet-LLP-fork.git
```

### Download the top-tagging datasets to /nfs/dust/

The full top tagging dataset can be found here:\
[https://zenodo.org/record/2603256](https://zenodo.org/record/2603256).

A smaller size version for training and validation is available here:\
[https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg](https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg).

To download the samples to your dust area on NAF, just do the following:

```bash
cd /nfs/dust/cms/user/<username>/ML_LLP
mkdir TopTaggingDataset #prepare a directory to store data
cd TopTaggingDataset
wget -O test.h5 'https://zenodo.org/record/2603256/files/test.h5'
wget -O train.h5 'https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg/download?path=%2Ftop-tagging&files=train.h5'
wget -O val.h5 'https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg/download?path=%2Ftop-tagging&files=val.h5'
```

### Loading a conda environment to jupyter notebooks

Launch JupyterHub at https://naf-jhub.desy.de/

Check `Select GPU node` and click on `Start`

On the browser, click on the directories to go to:

`nfs_dust/ML_LLP/ParticleNet-LLP-fork/tf-keras`

Here, you can find two jupyter notebooks:
- [convert_dataset.ipynb](tf-keras/convert_dataset.ipynb)
- [keras_train.ipynb](tf-keras/keras_train.ipynb)

**Note**: Every time you open a notebook, make sure to load your conda environment!\
You just need to click on `Kernel`, `Change kernel`, and `particlenet`.

### Run Keras/Tensorflow scripts

Convert the top tagging dataset with: [tf-keras/convert_dataset.ipynb](tf-keras/convert_dataset.ipynb)

Train the model with: [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb)

**Note:** Don't forget to change the `username` when needed!

------

# About ParticleNet

**Keras/TensorFlow implemetation**
 - [model](tf-keras/tf_keras_model.py)
 - Requires tensorflow>=2.0.0 or >=1.15rc2.
 - A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb).
    - The top tagging dataset can be obtained from [https://zenodo.org/record/2603256](https://zenodo.org/record/2603256) and converted with this [script](tf-keras/convert_dataset.ipynb).

## How to use the model

#### Keras/TensorFlow models

The ParticleNet model can be obtained by calling the `get_particle_net` function in [tf_keras_model.py](tf-keras/tf_keras_model.py), which takes three input arrays:
 - `points`: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
 - `features`: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
 - `mask`: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.

To have a simple implementation for batched training on GPUs, fixed-length input arrays are used for all the inputs, although in principle the  ParticleNet architecture can handle variable number of particles in each jet. Zero-padding is used for the `points` and `features` inputs such that they always have the same length, and a `mask` array is used to indicate if a position is occupied by a real particle or by a zero-padded value.

The implementation of a simplified model, ParticleNet-Lite, is also provided and can be accessed with the `get_particle_net_lite` function.

A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb).

## Citation
If you use ParticleNet in your research, please cite the paper:

	@article{Qu:2019gqs,
	      author         = "Qu, Huilin and Gouskos, Loukas",
	      title          = "{ParticleNet: Jet Tagging via Particle Clouds}",
	      year           = "2019",
	      eprint         = "1902.08570",
	      archivePrefix  = "arXiv",
	      primaryClass   = "hep-ph",
	      SLACcitation   = "%%CITATION = ARXIV:1902.08570;%%"
	}

## Acknowledgement
The ParticleNet model is developed based on the [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) model. The implementation of the EdgeConv operation in MXNet is adapted from the author's TensorFlow [implementation](https://github.com/WangYueFt/dgcnn), and also inspired by the MXNet [implementation](https://github.com/chinakook/PointCNN.MX) of PointCNN.

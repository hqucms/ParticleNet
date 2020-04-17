# ParticleNet-LLP-fork

Implementation of the jet classification network in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570).

Instructions to use ParticleNet architecture with naf-gpu infrastructure at DESY.

## Installing required packages

The full installation will take a while. Please be patient and carefully follow these instructions.

### Login

We will need to use naf-gpu login node. <username> is your personal username on naf.

```
ssh -XY <username>@naf-cms-gpu01.desy.de
```

### Create a link to /nfs/dust areas

This is needed to be able to see /nfs/dust files on naf-jhub.

```
cd ~
ln -s /nfs/dust/cms/user/<username> link_nfs_dust
```

### Create environment

We will install all the needed packages with anaconda.

```
# load cuda module
module load cuda
 
# change directory to your home directory on dust
cd /nfs/dust/cms/user/<username>/
 
# install anaconda
# ATTENTION! When asked where to install anaconda, do NOT simply approve by typing "y", but
# provide your dust home directory instead (type: /nfs/dust/cms/user/<username>/anaconda2).
#
# Answer all other upcoming prompts with the recommended option (emphasized by rectangular parentheses)
#
# Optional: If you want an easier way of activating your conda environment in the terminal later,
# you can allow anaconda to edit your .bashrc-file. You can also edit it later with the code snippet
# provided below.
wget https://repo.continuum.io/archive/Anaconda2-2019.10-Linux-x86_64.sh
bash Anaconda2-2019.10-Linux-x86_64.sh
# NOTE: The anaconda versions get updated from time to time. To have the latest version, you can look up the list at https://repo.continuum.io/archive/
# ALSO NOTE: The message about where to install Anaconda2 will be displayed after having read the license terms.
 
#OPTIONAL Edit your .bashrc file after installation.
gedit ~/.bashrc
# copy/paste this block somewhere in the file and replace "username" with your username:
 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nfs/dust/cms/user/<username>/anaconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nfs/dust/cms/user/<username>/anaconda2/etc/profile.d/conda.sh" ]; then
        . "/nfs/dust/cms/user/<username>/anaconda2/etc/profile.d/conda.sh"
    else
        export PATH="/nfs/dust/cms/user/<username>/anaconda2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
 
 
# after installation, load anaconda
# IMPORTANT NOTE!!! Once you finish this setup, you have to run the next command EVERY TIME you re-login into naf (except you chose to edit your .bashrc)!
export PATH=/nfs/dust/cms/user/<username>/anaconda2/bin:$PATH
 
# Now we create a conda environment. We are uncreative and call it "myenv"
conda create -n myenv python=3.6
source activate /nfs/dust/cms/user/<username>/anaconda2/envs/myenv
# IMPORTANT: In addition to the export PATH command, you will also need to re-run this command each time you login to naf (except you chose to edit your .bashrc)!
 
# NOTE: If you chose to edit your .bashrc yourself or have conda edit it in the installation process, you can activate your environment with these simple commands
# (needs to be run every time you re-login to NAF):
source .bashrc
source activate myenv
# note that source .bashrc is also done if you just run "bash"
 
# cd to your environment directory
# (note: this is IMPORTANT! If you try to install packages when not in your environment directory, you might get file I/O errors!)
cd /nfs/dust/cms/user/<username>/anaconda2/envs/myenv/
 
 
# Now we install some additional packages. Note that when installing packages with conda, the "solving environment" step can take forever.
# This is normal behavior so do not abort the installation (unless it runs longer than several hours).
# At first, install keras
conda install -c anaconda keras-gpu
 
# install tensorflow
conda install tensorflow
 
#install pandas (for data manipulation and analysis)
conda install pandas
 
# install matplotlib
conda install matplotlib
 
# install pytables
conda install pytables

# install ROOT
conda install -c conda-forge root
 
# install root_numpy
conda install -c conda-forge root_numpy

# install awkward
conda install -c conda-forge awkward

# install uproot_methods
conda install -c conda-forge uproot-methods

# install jupyterhub kernel
cd /nfs/dust/cms/user/<username>/anaconda2/envs/myenv #you should be here already, better to be sure
conda activate myenv
pip install ipykernel --user
python -m ipykernel install --user --name="myenv"
```

### Clone repository

If you followed these instructions, you should have tensorflow2.0 installed. Let's now clone this repository.

```
cd /nfs/dust/cms/user/<username>
mkdir ML_LLP #this is just my personal choice
cd ML_LLP
git clone https://github.com/lbenato/ParticleNet-LLP-fork.git

mkdir TopTaggingDataset #prepare a directory to store data
```

### Download and scp top tagging datasets on /nfs/dust

The full top tagging dataset can be found here: [https://zenodo.org/record/2603256](https://zenodo.org/record/2603256).

A smaller size version for training and validation is available here (consider only train.h5 and val.h5): [https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg](https://desycloud.desy.de/index.php/s/rKrtHqbQwb5TAfg).

Please download train, val and test in your PC. Then open a new shell on a terminal, and type:

```
scp <file_you_want_to_copy> <username>@naf-cms.desy.de:/nfs/dust/cms/user/<username>/ML_LLP/TopTaggingDataset/.
```

### Load environment to jupyter notebooks

Launch https://naf-jhub.desy.de/ and click on "Select GPU node", "Start". Click on "link_nfs_dust", "ML_LLP" and "ParticleNet-LLP-fork". Choose "tf-keras" and "convert_dataset.ipynb".

First important thing to do: click on "Kernel", "Change kernel", and choose "myenv". In this way, you have successfully loaded your conda environment. Remember to check this everytime you launch a notebook.


### Run Keras/Tensorflow scripts

Convert the top tagging dataset with: [tf-keras/convert_dataset.ipynb](tf-keras/convert_dataset.ipynb)
Train the model with: [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb)


------
**Keras/TensorFlow implemetation** 
 - [model](tf-keras/tf_keras_model.py)
 - Requires tensorflow>=2.0.0 or >=1.15rc2. 
 - A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb). 
    - The top tagging dataset can be obtained from [https://zenodo.org/record/2603256](https://zenodo.org/record/2603256) and converted with this [script](tf-keras/convert_dataset.ipynb). 

## How to use the model

#### Keras/TensorFlow models

The use of the Keras/TensorFlow model is similar to the MXNet model. A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb).

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

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

### Create environment

### Load environment to jupyter notebooks

### Download and scp top tagging datasets on /nfs/dust

### Run Keras/Tensorflow scripts

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

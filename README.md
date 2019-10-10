# ParticleNet

Implementation of the jet classification network in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570).

------

**MXNet implemetation**
 - [model](mxnet/particle_net.py)


**[New] Keras/TensorFlow implemetation** 
 - [model](tf-keras/tf_keras_model.py)
 - Requires tensorflow>=2.0.0 or >=1.15rc2. 
 - A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb). 
    - The top tagging dataset can be obtained from [https://zenodo.org/record/2603256](https://zenodo.org/record/2603256) and converted with this [script](tf-keras/convert_dataset.ipynb). 

## How to use the model

#### MXNet model

The ParticleNet model can be obtained by calling the `get_particle_net` function in [particle_net.py](mxnet/particle_net.py), which can return either an MXNet `Symbol` or an MXNet Gluon `HybridBlock`. The model takes three input arrays:
 - `points`: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
 - `features`: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
 - `mask`: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.

To have a simple implementation for batched training on GPUs, we use fixed-length input arrays for all the inputs, although in principle the  ParticleNet architecture can handle variable number of particles in each jet. Zero-padding is used for the `points` and `features` inputs such that they always have the same length, and a `mask` array is used to indicate if a position is occupied by a real particle or by a zero-padded value.

The implementation of a simplified model, ParticleNet-Lite, is also provided and can be accessed with the `get_particle_net_lite` function.

#### Keras/TensorFlow model

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

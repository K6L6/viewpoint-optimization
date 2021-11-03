# Added Files Explanation
train_mGPU.py is the main file to run for training
model_chain.py is the GQN model converted into chainer.Chain  
multi_process_updater.py contains training updater built from Chainer API for the purpose of running GQN on multi-gpu  
train_mGPU.py is the GQN training script modified to run with multi-gpu  
train_mGPU_extensions.py contains chainer extensions used in the training script (i.e. validation, learning rate annealing)  
observation4mgpu.py is a script which renders predictions after training GQN  

# Originally pulled from (https://github.com/musyoku/chainer-gqn.git)
# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

# Requirements (already included and installed by Docker file)

- Python 3
- h5py
- Chainer
    - `pip3 install chainer`
- CuPy
    - `pip3 install cupy-cuda100` for CUDA 10.0
    - `pip3 install cupy-cuda91` for CUDA 9.1

Also you need the followings for visualization:

- ffmpeg
    - `sudo apt install ffmpeg`
- imagemagick
    - `sudo apt install imagemagick`

# Network Architecture

![gqn_conv_draw](https://user-images.githubusercontent.com/15250418/50375239-ad31bb00-063d-11e9-9c1b-151c18dc265d.png)

![gqn_representation](https://user-images.githubusercontent.com/15250418/50375240-adca5180-063d-11e9-8b2a-fb2c3995bc33.png)

# Dataset

## deepmind/gqn-datasets

Datasets used to train GQN in the paper are available to download.

https://github.com/deepmind/gqn-datasets

You need to convert `.tfrecord` files to HDF5 `.h5` format before starting training.

https://github.com/musyoku/gqn-datasets-translator


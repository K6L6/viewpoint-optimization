# Use an official Ubuntu as a parent image
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Set the working directory to /GQN
WORKDIR /GQN

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt install -y software-properties-common\ 
			libopenmpi-dev\
			imagemagick\ 
			wget\ 
			mesa-utils\
			pkg-config\
			git-core\
			ffmpeg\
			x264

RUN useradd -m -r -u 17001 -g root lukman
USER lukman
WORKDIR /home/lukman

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh &&\
	bash miniconda.sh -b -p ./miniconda

RUN . ./miniconda/bin/activate && conda update -y conda
RUN export PATH=/home/lukman/miniconda/bin:$PATH

RUN . ./miniconda/bin/activate && conda install -y matplotlib \
	numpy \
	h5py \
	pandas \
	opencv

RUN . ./miniconda/bin/activate && pip install chainer\
		    cupy-cuda100\
			tensorflow-gpu\
			tensorflow more-itertools\
		    argparse\
		    uuid\
		    tabulate\
		    mpi4py\
		    # chainermn\
		    pybind11 \
			ipdb \
			gsutil \
			scikit-image \
			tensorboardx
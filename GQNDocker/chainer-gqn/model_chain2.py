#!/usr/bin/enc python2
import os
import sys
import uuid
import random
import ipdb

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy as cp
import numpy as numpy

from chainer.backends import cuda
from chainer.initializers import HeNormal
from chainer.serializers import save_hdf5
from trainer.scheduler import PixelVarianceScheduler

import gqn
from gqn import to_device
from gqn.preprocessing import preprocess_images
from hyperparams import HyperParameters

def encode_scene(images, viewpoints, model, gpu_device)
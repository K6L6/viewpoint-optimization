import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chaienr.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer. training import extensions

#input images ----
data_0 = np.array([5],dtype=np.float32)
x = Variable(data_0)
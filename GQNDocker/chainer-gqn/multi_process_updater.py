import multiprocessing
import warnings
import six
import math
import chainer
from chainer import reporter
import random
# import numpy as np
import cupy as cp
import logging
import ipdb
import time

import chainer.functions as cf

from chainer.training import updaters
from chainer.backends import cuda
from cupy.cuda import nccl
from chainer.training.updaters.multiprocess_parallel_updater import gather_grads,gather_params,scatter_grads,scatter_params,_get_nccl_data_type

from gqn import to_device
from gqn.preprocessing import preprocess_images
#==============================================================================
# Algorithms
#==============================================================================
def encode_scene(images, viewpoints, model, gpu_device):
    # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
    images = images.transpose((0, 1, 4, 2, 3)).astype(cp.float32)

    # Sample number of views
    total_views = images.shape[1]
    num_views = random.choice(range(1, total_views + 1))

    # Sample views
    observation_view_indices = list(range(total_views))
    random.shuffle(observation_view_indices)
    observation_view_indices = observation_view_indices[:num_views]

    observation_images = preprocess_images(
        images[:, observation_view_indices])

    observation_query = viewpoints[:, observation_view_indices]
    representation = model.compute_observation_representation(
        observation_images, observation_query)

    # Sample query view
    query_index = random.choice(range(total_views))
    query_images = preprocess_images(images[:, query_index])
    query_viewpoints = viewpoints[:, query_index]

    # Transfer to gpu if necessary
    query_images = to_device(query_images, gpu_device)
    query_viewpoints = to_device(query_viewpoints, gpu_device)

    return representation, query_images, query_viewpoints

def estimate_ELBO(xp, query_images, z_t_param_array, pixel_mean,
                    pixel_log_sigma, batch_size):
    # KL Diverge, pixel_ln_varnce
    kl_divergence = 0
    for params_t in z_t_param_array:
        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params_t
        normal_q = chainer.distributions.Normal(
            mean_z_q, log_scale=ln_var_z_q)
        normal_p = chainer.distributions.Normal(
            mean_z_p, log_scale=ln_var_z_p)
        kld_t = chainer.kl_divergence(normal_q, normal_p)
        kl_divergence += cf.sum(kld_t)
    kl_divergence = kl_divergence / batch_size

    # Negative log-likelihood of generated image
    batch_size = query_images.shape[0]
    num_pixels_per_batch =cp.prod(query_images.shape[1:])
    normal = chainer.distributions.Normal(
        query_images, log_scale=xp.array(pixel_log_sigma))
    
    log_px = cf.sum(normal.log_prob(pixel_mean)) / batch_size
    negative_log_likelihood = -log_px

    # Empirical ELBO
    ELBO = log_px - kl_divergence

    # https://arxiv.org/abs/1604.08772 Section.2
    # https://www.reddit.com/r/MachineLearning/comments/56m5o2/discussion_calculation_of_bitsdims/
    bits_per_pixel = -(ELBO / num_pixels_per_batch - cp.log(256)) / cp.log(
        2)

    return ELBO, bits_per_pixel, negative_log_likelihood, kl_divergence

class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = master.model
        self.device_index = master._devices[proc_id]
        self.device = chainer.get_device('@cupy:'+str(self.device_index))
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)
        self.pixel_log_sigma = master.pixel_log_sigma

    def setup(self):
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_device(self.device)
        self.reporter = reporter.Reporter()
        self.reporter.add_observer('main', self.model)
        self.reporter.add_observers('main',
                                    self.model.namedlinks(skipself=True))

    def run(self):
        self.device.use()

        self.setup()

        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                self.device.device.synchronize()
                break
            if job == 'update':
                # For reducing memory
                self.model.cleargrads()

                model = self.model
                model.cleargrads()

                x = self.converter(self.iterator.next(), self.device_index)
                batch_size = len(x)
                images = x['image']
                viewpoints = x['viewpoint']
                xp = model.xp

                representation, query_images, query_viewpoints = encode_scene(images, viewpoints, model, self.device_index)

                with self.reporter.scope({}):  # pass dummy observation
                    # Compute distribution parameterws
                    (z_t_param_array,
                        pixel_mean) = model.sample_z_and_x_params_from_posterior(
                            query_images, query_viewpoints, representation)

                    # Compute ELBO
                    (ELBO, bits_per_pixel, negative_log_likelihood,
                        kl_divergence) = estimate_ELBO(xp, query_images, z_t_param_array,
                                                    pixel_mean, self.pixel_log_sigma, batch_size)
                    
                    # Update parameters                   
                    loss = -ELBO
                
                loss.backward()
                del loss

                gg = gather_grads(self.model)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl_data_type, nccl.NCCL_SUM, 0,
                                 null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type, 0,
                                null_stream.ptr)
                scatter_params(self.model, gp)
                del gp

class CustomUpdater(updaters.StandardUpdater):
    """Custom updater for chainer.
    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (int or dict): The destination device info to send variables. In the
            case of cpu or single gpu, `device=-1 or 0`, respectively.
            In the case of multi-gpu, `device={"main":0, "sub_1": 1, ...}`.
    """

    def __init__(self, train_iter, optimizer, converter, device, pixel_log_sigma, accum_grad=1, **kw):
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)
        self.start = True
        self.device = device
        self.pixel_log_sigma = pixel_log_sigma

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine for Custom Updater."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get batch and convert into variables
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        images = x['image']
        viewpoints = x['viewpoint']
        
        if self.start:
            optimizer.target.cleargrads()
            self.start = False
        xp = optimizer.target.xp
        batch_size = len(batch)
        
        representation, query_images, query_viewpoints = encode_scene(images, viewpoints, optimizer.target, self.device)
        # Compute distribution parameters
        (z_t_param_array,
            pixel_mean) = optimizer.target.sample_z_and_x_params_from_posterior(
                query_images, query_viewpoints, representation)

        # Compute ELBO
        (ELBO, bits_per_pixel, negative_log_likelihood,
            kl_divergence) = estimate_ELBO(xp, query_images, z_t_param_array,
                                        pixel_mean, self.pixel_log_sigma, batch_size)

        #------------------------------------------------------------------------------
        # Update parameters
        #------------------------------------------------------------------------------
        loss = -ELBO
        optimizer.target.cleargrads()
        loss.backward()  # Backprop
        
        optimizer.update()
        with chainer.no_backprop_mode():
            mean_squared_error = cf.mean_squared_error(
                query_images, pixel_mean)
        
        reporter.report({'loss': float(loss.data), 
                        'bits_per_pixel':float(bits_per_pixel.data), 
                        'NLL':float(negative_log_likelihood.data),
                        'MSE':float(mean_squared_error.data)},
                        optimizer.target)
        optimizer.target.cleargrads()  # Clear the parameter gradients

class CustomParallelUpdater(updaters.MultiprocessParallelUpdater):
    def __init__(self, iterator, optimizer, devices, converter, pixel_log_sigma, **kw):
        super(CustomParallelUpdater, self).__init__(
            iterator, optimizer, devices=devices, converter=converter)
        self.start = True
        self.devices = devices
        self.pixel_log_sigma = pixel_log_sigma
        self.iterator = self.get_iterator('main')
        self.optimizer = self.get_optimizer('main')
        self.model = self.optimizer.target
        self._devices = [device for device in list(devices.values())]
        # self._devices = [chainer.get_device('@cupy:'+str(device)) for device in list(devices.values())]

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        self.model.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with chainer.using_device(self._devices[0]):
            self.model.to_device(self._devices[0])
            if len(self._devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(('set comm_id', comm_id))
                self.comm = nccl.NcclCommunicator(
                    len(self._devices), comm_id, 0)

    def update_core(self):
        self.setup_workers()
        self._send_message(('update', None))

        with chainer.using_device(self._devices[0]):
            iterator = self.get_iterator('main')
            optimizer = self.get_optimizer('main')
            model = self.model

            batch = iterator.next()
            x = self.converter(batch, self._devices[0]) #how to split devices?

            images = x['image']
            viewpoints = x['viewpoint']

            if self.start:
                model.cleargrads()
                self.start = False
            xp = model.xp
            batch_size = len(batch)
            
            #  For reducing memory
            model.cleargrads()
            
            #------------------------------------------------------------------------------
            # Scene encoder
            #------------------------------------------------------------------------------
            representation, query_images, query_viewpoints = encode_scene(images, viewpoints, model, self._devices[0])

            #------------------------------------------------------------------------------
            # Compute empirical ELBO
            #------------------------------------------------------------------------------
            # Compute distribution parameterws
            (z_t_param_array,
                pixel_mean) = model.sample_z_and_x_params_from_posterior(
                    query_images, query_viewpoints, representation)

            # Compute ELBO
            (ELBO, bits_per_pixel, negative_log_likelihood,
                kl_divergence) = estimate_ELBO(xp, query_images, z_t_param_array,
                                            pixel_mean, self.pixel_log_sigma, batch_size)

            #------------------------------------------------------------------------------
            # Update parameters
            #------------------------------------------------------------------------------
            loss = -ELBO
            
            loss.backward()
            # if start_training: 
            #     g = chainer.computational_graph.build_computational_graph(pixel_mean)
            #     with open(os.path.join(args.snapshot_directory,'cg.dot'), 'w') as o:
            #         o.write(g.dump())
            #     start_training = False
            # exit(1)

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(model)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                nccl_data_type, nccl.NCCL_SUM,
                                0, null_stream.ptr)
                scatter_grads(model, gg)
                del gg

            optimizer.update()
            if self.comm is not None:
                gp = gather_params(model)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type,
                                0, null_stream.ptr)

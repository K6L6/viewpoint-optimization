import os
import sys
import uuid
import ipdb

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
import logging
from chainer.backends import cuda
from chainer.initializers import HeNormal
from chainer.serializers import save_hdf5
from tensorboardX import SummaryWriter

import gqn
from hyperparams import HyperParameters


##############################################################################
# Use representation vector to produce uncertainty values across viewpoints
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 #/ try interpolate between viewpoints, if ground truth viewpoints are different between scenes.
#/ Representation and Inference network from GQN used

class NormalDistribution(chainer.Chain):
    def __init__(self, z_channels, weight_initializer=None):
        super().__init__()
        with self.init_scope():
            self.conv = nn.Convolution2D(
                None,
                z_channels * 2,
                ksize=5,
                stride=1,
                pad=2,
                initialW=weight_initializer)

    def compute_parameter(self, h):
        param = self.conv(h)
        mean, ln_var = cf.split_axis(param, 2, axis=1)
        return mean, ln_var

    def sample(self, h):
        mean, ln_var = self.compute_parameter(h)
        return cf.gaussian(mean, ln_var)


class Model():
    def __init__(self, hyperparams: HyperParameters):
        assert isinstance(hyperparams, HyperParameters)
        self.num_layers = hyperparams.num_layers
        self.hyperparams = hyperparams
        self.parameters = chainer.ChainList()
        self.snapshot_filename = "model.hdf5"

        h_size = (hyperparams.image_size[0] // 4,
                  hyperparams.image_size[0] // 4)
        v_size = h_size

        if hyperparams.representation_architecture == "tower":
            r_size = h_size
        elif hyperparams.representation_architecture == "pool":
            r_size = (1, 1)
        else:
            raise NotImplementedError

        weight_initializer = chainer.initializers.GlorotNormal()

        #------------------------------------------------------------------------------
        # Generation network
        #------------------------------------------------------------------------------
        self.generation_cores = []
        with self.parameters.init_scope():
            # LSTM core
            num_cores = 1 if self.hyperparams.generator_share_core else self.num_layers
            for _ in range(num_cores):
                core = gqn.nn.GenerationCore(
                    h_channels=hyperparams.h_channels,
                    h_size=h_size,
                    r_channels=hyperparams.r_channels,
                    r_size=r_size,
                    u_channels=hyperparams.u_channels,
                    weight_initializer=weight_initializer)
                self.generation_cores.append(core)
                self.parameters.append(core)

            # z prior
            self.z_prior_distribution = NormalDistribution(
                z_channels=hyperparams.z_channels)
            self.parameters.append(self.z_prior_distribution)

            # 1x1 conv (u -> x)
            if hyperparams.u_channels == 3:
                self._map_u_x = None
            else:
                self._map_u_x = nn.Convolution2D(
                    hyperparams.u_channels,
                    3,
                    ksize=1,
                    stride=1,
                    pad=0,
                    initialW=weight_initializer)
                self.parameters.append(self._map_u_x)

        #------------------------------------------------------------------------------
        # Inference network
        #------------------------------------------------------------------------------
        self.inference_cores = []
        with self.parameters.init_scope():
            num_cores = 1 if self.hyperparams.inference_share_core else self.num_layers
            for t in range(num_cores):
                # LSTM core
                core = gqn.nn.InferenceCore(
                    h_channels=hyperparams.h_channels,
                    h_size=h_size,
                    r_channels=hyperparams.r_channels,
                    r_size=r_size,
                    u_channels=hyperparams.u_channels,
                    weight_initializer=weight_initializer)
                self.inference_cores.append(core)
                self.parameters.append(core)

            # z posterior
            self.z_posterior_distribution = NormalDistribution(
                z_channels=hyperparams.z_channels)
            self.parameters.append(self.z_posterior_distribution)

        #------------------------------------------------------------------------------
        # Representation network
        #------------------------------------------------------------------------------
        if hyperparams.representation_architecture == "tower":
            self.representation_network = gqn.nn.TowerNetwork(
                r_channels=hyperparams.r_channels,
                v_size=v_size,
                weight_initializer=weight_initializer)
            with self.parameters.init_scope():
                self.parameters.append(self.representation_network)

        if hyperparams.representation_architecture == "pool":
            self.representation_network = gqn.nn.PoolNetwork(
                r_channels=hyperparams.r_channels,
                r_size=r_size,
                v_size=v_size,
                weight_initializer=weight_initializer)
            with self.parameters.init_scope():
                self.parameters.append(self.representation_network)

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    @property
    def num_trainable_parameters(self):
        size = 0
        for param in self.parameters.params():
            size += param.data.size
        return size

    def load(self, snapshot_root_directory, epoch):
        model_path = os.path.join(snapshot_root_directory,
                                  self.snapshot_filename)
        try:
            if os.path.exists(model_path):
                print("loading {}".format(model_path))
                chainer.serializers.load_hdf5(model_path, self.parameters)
            return True
        except Exception as error:
            print(error)
        return False

    def save(self, snapshot_root_directory, epoch):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(
            os.path.join(snapshot_root_directory, tmp_filename),
            self.parameters)
        os.rename(
            os.path.join(snapshot_root_directory, tmp_filename),
            os.path.join(snapshot_root_directory, self.snapshot_filename))

    def generate_initial_state(self, batch_size, xp):
        hc_size = (self.hyperparams.image_size[0] // 4,
                   self.hyperparams.image_size[1] // 4)
        initial_h_gen = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + hc_size,
            dtype="float32")
        initial_c_gen = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + hc_size,
            dtype="float32")
        initial_u = xp.zeros(
            (
                batch_size,
                self.hyperparams.u_channels,
            ) + self.hyperparams.image_size,
            dtype="float32")
        initial_h_enc = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + hc_size,
            dtype="float32")
        initial_c_enc = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + hc_size,
            dtype="float32")
        return initial_h_gen, initial_c_gen, initial_u, initial_h_enc, initial_c_enc

    def get_generation_core(self, t):
        if self.hyperparams.generator_share_core:
            return self.generation_cores[0]
        return self.generation_cores[t]

    def get_generation_prior(self, t):
        if self.hyperparams.generator_share_prior:
            return self.generation_priors[0]
        return self.generation_priors[t]

    def get_generation_upsampler(self, t):
        if self.hyperparams.generator_share_upsampler:
            return self.generation_upsamplers[0]
        return self.generation_upsamplers[t]

    def get_inference_core(self, t):
        if self.hyperparams.inference_share_core:
            return self.inference_cores[0]
        return self.inference_cores[t]

    def get_inference_posterior(self, t):
        if self.hyperparams.inference_share_posterior:
            return self.inference_posteriors[0]
        return self.inference_posteriors[t]

    def compute_observation_representation(self, images, viewpoints):
        batch_size = images.shape[0]
        num_views = images.shape[1]

        # (batch, views, channels, height, width) -> (batch * views, channels, height, width)
        images = images.reshape((batch_size * num_views, ) + images.shape[2:])
        viewpoints = viewpoints.reshape((batch_size * num_views, 7, 1, 1))

        # Transfer to gpu
        xp = self.parameters.xp
        if xp is cupy:
            images = cuda.to_gpu(images)
            viewpoints = cuda.to_gpu(viewpoints)

        # Add noise
        # images += xp.random.uniform(
        #     0, 1.0 / 256.0, size=images.shape).astype(xp.float32)

        r = self.representation_network(images, viewpoints)

        # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
        r = r.reshape((batch_size, num_views) + r.shape[1:])

        # Sum element-wise across views
        r = cf.sum(r, axis=1)

        return r

    def map_u_x(self, x):
        if self._map_u_x is None:
            return x
        return self._map_u_x(x)

    def sample_z_and_x_params_from_posterior(self, x, v, r):
        batch_size = x.shape[0]
        xp = cuda.get_array_module(x)

        h_t_gen, c_t_gen, u_t, h_t_enc, c_t_enc = self.generate_initial_state(
            batch_size, xp)
        v = cf.reshape(v, v.shape + (1, 1))

        z_t_params_array = []

        for t in range(self.num_layers):
            inference_core = self.get_inference_core(t)
            generation_core = self.get_generation_core(t)

            h_next_enc, c_next_enc = inference_core(h_t_gen, h_t_enc, c_t_enc,
                                                    x, v, r, u_t)

            # obtain the mean and var from below for observation
            mean_z_q, ln_var_z_q = self.z_posterior_distribution.compute_parameter(
                h_t_enc)
            z_t = cf.gaussian(mean_z_q, ln_var_z_q)

            mean_z_p, ln_var_z_p = self.z_prior_distribution.compute_parameter(
                h_t_gen)

            h_next_gen, c_next_gen, u_next = generation_core(
                h_t_gen, c_t_gen, z_t, v, r, u_t)

            z_t_params_array.append((mean_z_q, ln_var_z_q, mean_z_p,
                                     ln_var_z_p))

            u_t = u_next
            u_t.name = 'u_{}'.format(t)
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen
            h_t_enc = h_next_enc
            c_t_enc = c_next_enc

        mean_x = self.map_u_x(u_t)
        return z_t_params_array, mean_x

    def generate_image(self, v, r):
        xp = cuda.get_array_module(v)

        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)
        v = cf.reshape(v, v.shape[:2] + (1, 1))

        for t in range(self.num_layers):
            generation_core = self.get_generation_core(t)

            mean_z_p, ln_var_z_p = self.z_prior_distribution.compute_parameter(
                h_t_gen)
            z_t = cf.gaussian(mean_z_p, ln_var_z_p)

            h_next_gen, c_next_gen, u_next = generation_core(
                h_t_gen, c_t_gen, z_t, v, r, u_t)

            u_t = u_next
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen

        mean_x = self.map_u_x(u_t)
        return mean_x.data, ln_var_z_p, z_t

    def generate_multi_image(self, v, r, no_of_samples):
        '''used to generate predicted images from z values sampled from a Gaussian Dist.'''
        writer = SummaryWriter('/GQN/chainer-gqn/tensor-log')
        xp = cuda.get_array_module(v)

        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)
        v = cf.reshape(v, v.shape[:2] + (1, 1))

        # no_of_samples = 100
        reconstructed_images = []
        
        # ht_list = []
        # ut_list = []
        for i in range(no_of_samples):
            h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)
            for t in range(self.num_layers):
                generation_core = self.get_generation_core(t)

                mean_z_p, ln_var_z_p = self.z_prior_distribution.compute_parameter(
                    h_t_gen)
                z_t = cf.gaussian(mean_z_p, ln_var_z_p)
                writer.add_histogram('Variance of Z',cp.mean(cp.var(z_t,axis=0)).data,t)
                logging.INFO("logged variance of Z")

                h_next_gen, c_next_gen, u_next = generation_core(
                    h_t_gen, c_t_gen, z_t, v, r, u_t)

                u_t = u_next
                h_t_gen = h_next_gen
                c_t_gen = c_next_gen
                writer.add_histogram('Variance of Predicted images',u_t.data,t)
                logging.INFO("logged variance of predicted image")
                writer.close()
            mean_x = self.map_u_x(u_t)
            # reconstructed_images.append(mean_x.data)
            if i == 0:
                shapes_c = [no_of_samples] + [x for x in c_t_gen.shape]
                shapes_z = [no_of_samples] + [x for x in z_t.shape]
                shapes_images = [no_of_samples] + [x for x in mean_x.shape]
                ct_list = cupy.zeros(shapes_c)
                zt_list = cupy.zeros(shapes_z)
                reconstructed_images = cupy.zeros(shapes_images)
            zt_list[i] = z_t.data
            ct_list[i] = c_t_gen.data
            reconstructed_images[i] = mean_x.data 
            # ht_list.append(h_t_gen.data)
            # ut_list.append(u_t.data)
            
        return reconstructed_images, ln_var_z_p.data, zt_list, ct_list
        # , ht_list, ut_list

    def generate_image_from_zero_z(self, v, r):
        xp = cuda.get_array_module(v)

        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)

        v = cf.reshape(v, v.shape[:2] + (1, 1))

        for t in range(self.num_layers):
            generation_core = self.get_generation_core(t)

            mean_z_p, _ = self.z_prior_distribution.compute_parameter(h_t_gen)
            z_t = xp.zeros_like(mean_z_p.data)

            h_next_gen, c_next_gen, u_next = generation_core(
                h_t_gen, c_t_gen, z_t, v, r, u_t)

            u_t = u_next
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen

        mean_x = self.map_u_x(u_t)
        return mean_x.data

    def generate_canvas_states(self, v, r, xp):
        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)

        v = cf.reshape(v, v.shape[:2] + (1, 1))

        u_t_array = []

        for t in range(self.num_layers):
            generation_core = self.get_generation_core(t)

            mean_z_p, ln_var_z_p = self.z_prior_distribution.compute_parameter(
                h_t_gen)
            z_t = cf.gaussian(mean_z_p, ln_var_z_p)

            h_next_gen, c_next_gen, u_next = generation_core(
                h_t_gen, c_t_gen, z_t, v, r, u_t)

            u_t = u_next
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen

            u_t_array.append(u_t)

        return u_t_array

    def reconstruct_image(self, query_images, v, r, xp):
        batch_size = v.shape[0]
        h0_g, c0_g, u0, h0_e, c0_e = self.generate_initial_state(
            batch_size, xp)
        v = cf.reshape(v, v.shape[:2] + (1, 1))

        hl_e = h0_e
        cl_e = c0_e
        hl_g = h0_g
        cl_g = c0_g
        ul_e = u0

        for l in range(self.num_layers):
            inference_core = self.get_inference_core(l)
            inference_posterior = self.get_inference_posterior(l)
            generation_core = self.get_generation_core(l)

            he_next, ce_next = inference_core(hl_g, hl_e, cl_e, x, v, r, ul_e)

            ze_l = inference_posterior.sample_z(hl_e)

            hg_next, cg_next, ue_next = generation_core(
                hl_g, cl_g, ul_e, ze_l, v, r)

            hl_g = hg_next
            cl_g = cg_next
            ul_e = ue_next
            hl_e = he_next
            cl_e = ce_next

        x = self.generation_observation.compute_mean_x(ul_e)
        return x.data


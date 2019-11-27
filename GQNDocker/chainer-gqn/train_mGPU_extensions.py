import math
import ipdb
import chainer
import numpy as np
import chainer.functions as cf
import pandas as pd

from chainer.training.extensions import Evaluator

from model_chain import encode_scene,estimate_ELBO
from chainer import reporter
from chainer.training import extension

from sys import stdout

class AnnealLearningRate(extension.Extension):
    def __init__(self,initial_lr,final_lr,annealing_steps,optimizer):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.annealing_steps = annealing_steps
        self.last_alpha = None
        self.t = 0

    def initialize(self,trainer):
        pass

    def compute_lr_at_step(self,training_step):
        return max(
            self.final_lr + (self.initial_lr - self.final_lr) *
            (1.0 - training_step / self.annealing_steps), self.final_lr)
        
    def __call__(self,trainer):
        self.t +=1
        optimizer = self.optimizer
        new_alpha = self.compute_lr_at_step(
            self.t)
        self.update_alpha(optimizer, new_alpha)

    def update_alpha(self, optimizer, new_alpha):
        setattr(optimizer, 'alpha',new_alpha)
        self.last_alpha = new_alpha

class Validation(Evaluator):
    def __init__(self, iterator, converter, model, variance_scheduler, device):
        super(Validation, self).__init__(iterator, model)
        self.iterator = iterator
        self.converter = converter
        self.model = model
        self.device = device
        self.variance_scheduler = variance_scheduler
    
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter.DictSummary()
        xp = self.model.xp
        with chainer.no_backprop_mode():

            for batch in it:
                observation = {}
                data = self.converter(batch, self.device)
                batch_size = len(batch)
                pixel_log_sigma_test = xp.full(
                        (batch_size, 3) + self.model.hyperparams.image_size,
                        math.log(self.variance_scheduler.standard_deviation),
                        dtype="float32")
                images = data['image']
                viewpoints = data['viewpoint']
                with reporter.report_scope(observation):
                    # Scene encoder
                    representation, query_images, query_viewpoints = encode_scene(
                        images, viewpoints, self.model, self.device)
                    
                    # Compute empirical ELBO
                    (z_t_param_array, pixel_mean
                        ) = self.model.sample_z_and_x_params_from_posterior(
                            query_images, query_viewpoints, representation)
                    (ELBO, bits_per_pixel, NLL, KLD) = estimate_ELBO(
                            xp, query_images, z_t_param_array, pixel_mean, pixel_log_sigma_test, batch_size)
                    MSE = cf.mean_squared_error(
                        query_images, pixel_mean)

                    reporter.report({
                        'ELBO':float(ELBO.data),
                        'bits_per_pixel':float(bits_per_pixel.data), 
                        'NLL':float(NLL.data),
                        'KLD':float(KLD.data),
                        'MSE':float(MSE.data)},
                        self.model)
                summary.add(observation)
            
            # reporter.report({
            #                 'ELBO':float(ELBO.data),
            #                 'bits_per_pixel':float(bits_per_pixel.data),
            #                 'NLL':float(negative_log_likelihood.data),
            #                 'KLD':float(kl_divergence.data),
            #                 'MSE':float(mean_squared_error.data)
            #                 })
        return summary.compute_mean()
    
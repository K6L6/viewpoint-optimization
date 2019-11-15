import argparse
import math
import os
import random
import sys

import chainer
import chainer.functions as cf
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda
from chainer.training import extensions
from multi_process_updater import *
import chainerx
import logging
from train_mGPU_extensions import AnnealLearningRate, Validation

import gqn
from gqn import to_device
from gqn.data import Dataset, Iterator
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from model_chain import Model
from trainer.dataframe import DataFrame
from trainer.meter import Meter
from trainer.optimizer import AdamOptimizer
from trainer.scheduler import PixelVarianceScheduler

import ipdb

def _mkdir(directory):
    try:
        os.makedirs(directory)
    except:
        pass

def main():

    os.environ['CHAINER_SEED'] = str(args.seed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    _mkdir(args.snapshot_directory)
    _mkdir(args.log_directory)

    meter_train = Meter()
    meter_train.load(args.snapshot_directory)

    #==============================================================================
    # Dataset
    #==============================================================================
    def read_files(directory):
        filenames = []
        files = os.listdir(os.path.join(directory, "images"))
        for filename in files:
            if filename.endswith(".npy"):
                filenames.append(filename)
        filenames.sort()
        
        dataset_images = []
        dataset_viewpoints = []
        for i in range(len(filenames)):
            images_npy_path = os.path.join(directory, "images", filenames[i])
            viewpoints_npy_path = os.path.join(directory, "viewpoints", filenames[i])
            tmp_images = np.load(images_npy_path)
            tmp_viewpoints = np.load(viewpoints_npy_path)
        
            assert tmp_images.shape[0] == tmp_viewpoints.shape[0]
            
            dataset_images.extend(tmp_images)
            dataset_viewpoints.extend(tmp_viewpoints)
        dataset_images = np.array(dataset_images)
        dataset_viewpoints = np.array(dataset_viewpoints)

        dataset = list()
        for i in range(len(dataset_images)):
            item = {'image':dataset_images[i],'viewpoint':dataset_viewpoints[i]}
            dataset.append(item)
        
        return dataset

    dataset_train = read_files(args.train_dataset_directory)
    if args.test_dataset_directory is not None:
        dataset_test = read_files(args.test_dataset_directory)
    
    # ipdb.set_trace()
    
    #==============================================================================
    # Hyperparameters
    #==============================================================================
    hyperparams = HyperParameters()
    hyperparams.num_layers = args.generation_steps
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.h_channels = args.h_channels
    hyperparams.z_channels = args.z_channels
    hyperparams.u_channels = args.u_channels
    hyperparams.r_channels = args.r_channels
    hyperparams.image_size = (args.image_size, args.image_size)
    hyperparams.representation_architecture = args.representation_architecture
    hyperparams.pixel_sigma_annealing_steps = args.pixel_sigma_annealing_steps
    hyperparams.initial_pixel_sigma = args.initial_pixel_sigma
    hyperparams.final_pixel_sigma = args.final_pixel_sigma

    hyperparams.save(args.snapshot_directory)
    print(hyperparams, "\n")

    #==============================================================================
    # Model
    #==============================================================================
    model = Model(hyperparams)
    model.load(args.snapshot_directory, meter_train.epoch)
    
    #==============================================================================
    # Pixel-variance annealing
    #==============================================================================
    variance_scheduler = PixelVarianceScheduler(
        sigma_start=args.initial_pixel_sigma,
        sigma_end=args.final_pixel_sigma,
        final_num_updates=args.pixel_sigma_annealing_steps)
    variance_scheduler.load(args.snapshot_directory)
    print(variance_scheduler, "\n")

    pixel_log_sigma = np.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(variance_scheduler.standard_deviation),
        dtype="float32")

    #==============================================================================
    # Selecting the GPU
    #==============================================================================
    # xp = np
    # gpu_device = args.gpu_device
    # using_gpu = gpu_device >= 0
    # if using_gpu:
    #     cuda.get_device(gpu_device).use()
    #     xp = cp

    # devices = tuple([chainer.get_device(f"@cupy:{gpu}") for gpu in args.gpu_devices])
    # if any(device.xp is chainerx for device in devices):
    #     sys.stderr.write("Cannot support ChainerX devices.")
    #     sys.exit(1)

    ngpu = args.ngpu
    using_gpu = ngpu > 0
    xp=cp
    if ngpu == 1:
        gpu_id = 0
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
        logging.info('single gpu calculation.')
    elif ngpu > 1:
        gpu_id = 0
        devices = {'main': gpu_id}
        for gid in six.moves.xrange(1, ngpu):
            devices['sub_%d' % gid] = gid
        logging.info('multi gpu calculation (#gpus = %d).' % ngpu)
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
    else:
        gpu_id = -1
        logging.info('cpu calculation')

    #==============================================================================
    # Logging
    #==============================================================================
    csv = DataFrame()
    csv.load(args.log_directory)

    #==============================================================================
    # Optimizer
    #==============================================================================
    initial_training_step=0
    # lr = compute_lr_at_step(initial_training_step) # function in GQN AdamOptimizer
    
    optimizer = chainer.optimizers.Adam(beta1=0.9, beta2=0.99, eps=1e-8) #lr is needed originally
    optimizer.setup(model)
    # optimizer = AdamOptimizer(
    #     model.parameters,
    #     initial_lr=args.initial_lr,
    #     final_lr=args.final_lr,
    #     initial_training_step=variance_scheduler.training_step)
    # )
    print(optimizer, "\n")


    #==============================================================================
    # Training iterations
    #==============================================================================
    if ngpu>1:

        train_iters = [
            chainer.iterators.MultiprocessIterator(dataset_train, args.batch_size, n_processes=args.number_processes) for i in chainer.datasets.split_dataset_n_random(dataset_train,len(devices))
        ]
        updater = CustomParallelUpdater(train_iters,optimizer,devices)
    
    elif ngpu==1:
        
        train_iters = chainer.iterators.SerialIterator(dataset_train,args.batch_size,shuffle=True)
        # ipdb.set_trace()
        updater = CustomUpdater(train_iters, optimizer, converter=chainer.dataset.concat_examples, device=0, pixel_log_sigma=pixel_log_sigma)
        
    else:
        raise NotImplementedError('Implement for single gpu or cpu')
    
    trainer = chainer.training.Trainer(updater,(args.epochs,'epoch'),args.snapshot_directory)
    trainer.extend(AnnealLearningRate(
                                    initial_lr=args.initial_lr,
                                    final_lr=args.final_lr,
                                    annealing_steps=args.pixel_sigma_annealing_steps,
                                    optimizer=optimizer),
                                    trigger=(1,'iteration'))
    # add information per epoch with report?
    # add learning rate annealing, snapshot saver, evaluator
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'))
    trainer.extend(extensions.ProgressBar())
    reports = ['main/loss', 'main/bits_per_pixel', 'main/NLL', 'main/MSE']
    #Validation
    if args.test_dataset_directory is not None:
        test_iters = chainer.iterators.SerialIterator(dataset_test,args.batch_size*6,shuffle=True)
        trainer.extend(Validation(test_iters,chainer.dataset.concat_examples,optimizer.target,variance_scheduler,device=0))
        reports.append('validation/main/loss')
        reports.append('validation/main/bits_per_pixel')
        reports.append('validation/main/NLL')
        reports.append('validation/main/MSE')
    trainer.extend(
        extensions.PrintReport(reports), trigger=(args.report_interval_iters, 'iteration'))
    

    # np.random.seed(args.seed)
    # cp.random.seed(args.seed)

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-directory", type=str, required=True)
    parser.add_argument("--test-dataset-directory", type=str, default=None)
    parser.add_argument("--snapshot-directory", type=str, default="snapshots")
    parser.add_argument("--log-directory", type=str, default="log")
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--ngpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--generation-steps", type=int, default=12)
    parser.add_argument("--initial-lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--initial-pixel-sigma", type=float, default=2.0)
    parser.add_argument("--final-pixel-sigma", type=float, default=0.7)
    parser.add_argument(
        "--pixel-sigma-annealing-steps", type=int, default=160000)
    parser.add_argument("--h-channels", type=int, default=128)
    parser.add_argument("--z-channels", type=int, default=3)
    parser.add_argument("--u-channels", type=int, default=128)
    parser.add_argument("--r-channels", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument(
        "--representation-architecture",
        type=str,
        default="tower",
        choices=["tower", "pool"])
    parser.add_argument("--generator-share-core", action="store_true")
    parser.add_argument("--inference-share-core", action="store_true")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--number-processes",type=int) # number of CPU cores to split preprocessing of data.
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--report-interval-iters", type=int, default=10)
    args = parser.parse_args()
    
    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    
    main()

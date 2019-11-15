import argparse
import math
import time
import sys
import os
import random
import operator
import ipdb

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import chainer.functions as cf
import cupy as cp
import numpy as np
from chainer.backends import cuda
from skimage.filters import threshold_otsu

sys.path.append("../../")
import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch
from model_testing import Model
# from model import Model
from trainer.meter import Meter


def main():
    try:
        os.makedirs(args.figure_directory)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    dataset = gqn.data.Dataset(args.dataset_directory, 
                                # use_ground_truth=True
                                )

    meter = Meter()
    assert meter.load(args.snapshot_directory)

    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)

    model = Model(hyperparams)
    assert model.load(args.snapshot_directory, meter.epoch)

    if using_gpu:
        model.to_gpu()

    total_observations_per_scene = 4
    fps = 30

    black_color = -0.5
    image_shape = (3, ) + hyperparams.image_size
    axis_observations_image = np.zeros(
        (3, image_shape[1], total_observations_per_scene * image_shape[2]),
        dtype=np.float32)

    #==============================================================================
    # Utilities
    #==============================================================================
    def to_device(array):
        if using_gpu:
            array = cuda.to_gpu(array)
        return array

    def fill_observations_axis(observation_images):
        axis_observations_image = np.full(
            (3, image_shape[1], total_observations_per_scene * image_shape[2]),
            black_color,
            dtype=np.float32)
        num_current_obs = len(observation_images)
        total_obs = total_observations_per_scene
        width = image_shape[2]
        x_start = width * (total_obs - num_current_obs) // 2
        for obs_image in observation_images:
            x_end = x_start + width
            axis_observations_image[:, :, x_start:x_end] = obs_image
            x_start += width
        return axis_observations_image

    def compute_camera_angle_at_frame(t):
        horizontal_angle_rad = 2 * t * math.pi / (fps * 2) + math.pi / 4
        y_rad_top = math.pi / 3
        y_rad_bottom = -math.pi / 3
        y_rad_range = y_rad_bottom - y_rad_top
        if t < fps * 1.5:
            vertical_angle_rad = y_rad_top
        elif fps * 1.5 <= t and t < fps * 2.5:
            interp = (t - fps * 1.5) / fps
            vertical_angle_rad = y_rad_top + interp * y_rad_range
        elif fps * 2.5 <= t and t < fps * 4:
            vertical_angle_rad = y_rad_bottom
        elif fps * 4.0 <= t and t < fps * 5:
            interp = (t - fps * 4.0) / fps
            vertical_angle_rad = y_rad_bottom - interp * y_rad_range
        else:
            vertical_angle_rad = y_rad_top
        return horizontal_angle_rad, vertical_angle_rad

    def rotate_query_viewpoint(horizontal_angle_rad, vertical_angle_rad,
                               camera_distance):
        camera_direction = np.array([
            math.sin(horizontal_angle_rad),  # x
            math.sin(vertical_angle_rad),  # y
            math.cos(horizontal_angle_rad),  # z
        ])
        camera_direction = camera_distance * camera_direction / np.linalg.norm(
            camera_direction)
        yaw, pitch = compute_yaw_and_pitch(camera_direction)
        query_viewpoints = xp.array(
            (
                camera_direction[0],
                camera_direction[1],
                camera_direction[2],
                math.cos(yaw),
                math.sin(yaw),
                math.cos(pitch),
                math.sin(pitch),
            ),
            dtype=np.float32,
        )
        query_viewpoints = xp.broadcast_to(query_viewpoints,
                                           (1, ) + query_viewpoints.shape)
        return query_viewpoints

# added/modified
    def compute_horizontal_rotation_at_frame(t):
        '''This rotates the scene horizontally.'''
        horizontal_angle_rad = 2 * t * math.pi / (fps * 2) + math.pi / 4
        vertical_angle_rad = 0
        
        return horizontal_angle_rad, vertical_angle_rad
    
    def get_mse_image(ground_truth,predicted):
        '''Calculates MSE between ground truth and predicted observation, and returns an image.'''
        assert ground_truth.shape==predicted.shape
    
        mse_image = np.square(ground_truth-predicted)*0.5
        mse_image = np.concatenate(mse_image).astype(np.float32)
        mse_image = np.reshape(mse_image,(3,64,64))
           
        return mse_image.transpose(1,2,0)

    def render(representation,
               camera_distance,
               start_t,
               end_t,
               gt_images,
               gt_viewpoints,
               animation_frame_array,
               rotate_camera=True):
        
        gt_images = np.squeeze(gt_images)
        gt_viewpoints = cp.reshape(cp.asarray(gt_viewpoints),(15,1,7))
        idx = cp.argsort(cp.squeeze(gt_viewpoints)[:,0])

        gt_images = [i for i, v in sorted(zip(gt_images, idx),key=operator.itemgetter(1))]
        gt_viewpoints = [i for i, v in sorted(zip(gt_viewpoints, idx),key=operator.itemgetter(1))]
        count = 0

        '''shows variance and mean images of 100 samples from the Gaussian.'''
        for t in range(start_t, end_t):
            artist_array = [
                axis_observations.imshow(
                    make_uint8(axis_observations_image),
                    interpolation="none",
                    animated=True)
            ]

            horizontal_angle_rad, vertical_angle_rad = compute_camera_angle_at_frame(
                t)

            if rotate_camera == False:
                horizontal_angle_rad, vertical_angle_rad = compute_camera_angle_at_frame(
                    0)
            query_viewpoints = rotate_query_viewpoint(
                horizontal_angle_rad, vertical_angle_rad, camera_distance)

            # shape 100x1x3x64x64, when Model is from model_testing.py
            generated_images = model.generate_image(query_viewpoints,
                                                    representation,100)
            
            # generate predicted from ground truth viewpoints
            predicted_images = model.generate_image(gt_viewpoints[count], representation,1)
            
            # predicted_images = model.generate_image(query_viewpoints, representation,1)
            predicted_images = np.squeeze(predicted_images)
            image_mse = get_mse_image(gt_images[count],predicted_images)
            
            # when sampling with 100
            cpu_generated_images = chainer.backends.cuda.to_cpu(generated_images)
            generated_images = np.squeeze(cpu_generated_images)
            
            # # cpu calculation
            # cpu_image_mean = np.mean(cpu_generated_images,axis=0)
            # cpu_image_std = np.std(cpu_generated_images,axis=0)
            # cpu_image_var = np.var(cpu_generated_images,axis=0) 
            # image_mean = np.squeeze(chainer.backends.cuda.to_gpu(cpu_image_mean))
            # image_std = chainer.backends.cuda.to_gpu(cpu_image_std)
            # image_var = np.squeeze(chainer.backends.cuda.to_gpu(cpu_image_var))
            
            image_mean = cp.mean(cp.squeeze(generated_images),axis=0)
            image_var = cp.var(cp.squeeze(generated_images),axis=0)

            # convert to black and white.
            # grayscale
            r,g,b = image_var
            gray_image_var = 0.2989*r+0.5870*g+0.1140*b            
            # thresholding Otsu's method
            thresh = threshold_otsu(gray_image_var)
            var_binary = gray_image_var > thresh
            
            sample_image = np.squeeze(generated_images[0])
            
            if count == 14:
                count = 0
            elif (t-fps)%10==0:
                count += 1
            
            print("computed an image. Count =",count)
            
            artist_array.append(
                axis_generation_variance.imshow(
                    var_binary,
                    cmap=plt.cm.gray,
                    interpolation="none",
                    animated=True))
            artist_array.append(
                axis_generation_mean.imshow(
                    make_uint8(image_mean),
                    interpolation="none",
                    animated=True))            
            artist_array.append(
                axis_generation_sample.imshow(
                    make_uint8(sample_image),
                    interpolation="none",
                    animated=True))
            artist_array.append(
                axis_generation_mse.imshow(
                    make_uint8(image_mse),
                    cmap='gray',
                    interpolation="none",
                    animated=True))

            animation_frame_array.append(artist_array)

    #==============================================================================
    # Visualization
    #==============================================================================
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(6, 7))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # fig.suptitle("GQN")
    axis_observations = fig.add_subplot(3, 1, 1)
    axis_observations.axis("off")
    axis_observations.set_title("observations")

    axis_generation_mse = fig.add_subplot(3, 2, 3)
    axis_generation_mse.axis("off")
    axis_generation_mse.set_title("MSE")

    axis_generation_variance = fig.add_subplot(3, 2, 4)
    axis_generation_variance.axis("off")
    axis_generation_variance.set_title("Variance")

    axis_generation_mean = fig.add_subplot(3, 2, 5)
    axis_generation_mean.axis("off")
    axis_generation_mean.set_title("Mean")

    axis_generation_sample = fig.add_subplot(3, 2, 6)
    axis_generation_sample.axis("off")
    axis_generation_sample.set_title("Normal Rendering")

    #==============================================================================
    # Generating animation
    #==============================================================================
    file_number = 1
    random.seed(0)
    np.random.seed(0)

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                animation_frame_array = []

                # shape: (batch, views, height, width, channels)
                images, viewpoints = subset[data_indices]
                # images, viewpoints, original images = subset[data_indices]
                camera_distance = np.mean(
                    np.linalg.norm(viewpoints[:, :, :3], axis=2))

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = preprocess_images(images)

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                # original_images = original_images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                # original_images = preprocess_images(original_images)

                batch_index = 0

                total_views = images.shape[1]
                random_observation_view_indices = list(range(total_views))
                random.shuffle(random_observation_view_indices)
                random_viewed_observation_indices = random_observation_view_indices[:
                                                                                  total_observations_per_scene]
                #------------------------------------------------------------------------------
                # Ground Truth
                #------------------------------------------------------------------------------
                
                gt_images = images
                gt_viewpoints = viewpoints
                # gt_images = original_images

                #------------------------------------------------------------------------------
                # Observations
                #------------------------------------------------------------------------------
                observed_images = images[batch_index,
                                         random_viewed_observation_indices]
                observed_viewpoints = viewpoints[
                    batch_index, random_viewed_observation_indices]

                observed_images = to_device(observed_images)
                observed_viewpoints = to_device(observed_viewpoints)

                #------------------------------------------------------------------------------
                # Generate images with a single observation
                #------------------------------------------------------------------------------
                # Scene encoder
                representation = model.compute_observation_representation(
                    observed_images[None, :1], observed_viewpoints[None, :1])

                # Update figure
                observation_index = random_viewed_observation_indices[0]
                observed_image = images[batch_index, observation_index]
                axis_observations_image = fill_observations_axis(
                    [observed_image])

                # Neural rendering
                render(representation, camera_distance, fps, fps * 6, gt_images, gt_viewpoints,
                       animation_frame_array)

                #------------------------------------------------------------------------------
                # Add observations
                #------------------------------------------------------------------------------
                for n in range(1, total_observations_per_scene):
                    observation_indices = random_viewed_observation_indices[:n +
                                                                          1]
                    axis_observations_image = fill_observations_axis(
                        images[batch_index, observation_indices])

                    # Scene encoder
                    representation = model.compute_observation_representation(
                        observed_images[None, :n + 1],
                        observed_viewpoints[None, :n + 1])

                    # Neural rendering
                    render(
                        representation,
                        camera_distance,
                        0,
                        fps // 2,
                        gt_images,
                        gt_viewpoints,
                        animation_frame_array,
                        rotate_camera=False)

                #------------------------------------------------------------------------------
                # Generate images with all observations
                #------------------------------------------------------------------------------
                # Scene encoder
                representation = model.compute_observation_representation(
                    observed_images[None, :total_observations_per_scene + 1],
                    observed_viewpoints[None, :total_observations_per_scene +
                                        1])

                # Neural rendering
                render(representation, camera_distance, 0, fps * 6, gt_images, gt_viewpoints,
                       animation_frame_array)

                #------------------------------------------------------------------------------
                # Write to file
                #------------------------------------------------------------------------------
                anim = animation.ArtistAnimation(
                    fig,
                    animation_frame_array,
                    interval=1 / fps,
                    blit=True,
                    repeat_delay=0)

                # anim.save(
                #     "{}/shepard_metzler_observations_{}.gif".format(
                #         args.figure_directory, file_number),
                #     writer="imagemagick",
                #     fps=fps)
                anim.save(
                    "{}/shepard_metzler_observations_{}.mp4".format(
                        args.figure_directory, file_number),
                    writer="ffmpeg",
                    fps=fps)

                print("video saved")
                file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dataset-directory", type=str, required=True)
    parser.add_argument("--snapshot-directory", "-snapshot", type=str, required=True)
    parser.add_argument("--figure-directory", type=str, required=True)
    args = parser.parse_args()
    main()

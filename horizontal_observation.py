import argparse
import math
import time
import sys
import os
import random
import ipdb
import h5py
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import chainer.functions as cf
import cupy as cp
import numpy as np
from chainer.backends import cuda
from tensorboardX import SummaryWriter

sys.path.append("../../")
import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch
from horiobs_model import Model
from trainer.meter import Meter


def main():
    start_time = time.time()
    
    writer = SummaryWriter('/GQN/chainer-gqn/tensor-log')

    try:
        os.makedirs(args.figure_directory)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    dataset = gqn.data.Dataset(args.dataset_directory)

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
    
    def compute_vertical_rotation_at_frame(horizontal,vertical,t):
        # move horizontal view only
        horizontal_angle_rad = horizontal + (t - fps) * (math.pi / 64)
        vertical_angle_rad = vertical + 0
    
        return horizontal_angle_rad, vertical_angle_rad

    def rotate_query_viewpoint(horizontal_angle_rad, vertical_angle_rad,
                               camera_distance):
        camera_direction = np.array([
            math.sin(horizontal_angle_rad),  # x
            math.sin(vertical_angle_rad),  # y
            math.cos(horizontal_angle_rad),  # z
            
        ])
        
        # removed linalg norm for observation purposes
        camera_direction = camera_distance * camera_direction
        # ipdb.set_trace()
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

    def render(representation,
               camera_distance,
               obs_viewpoint,
               start_t,
               end_t,
               animation_frame_array,savename=None,
               rotate_camera=True):
        
        all_var_bg = []
        all_var = []
        all_var_z = []
        all_q_view = []

        all_c = []
        all_h = []
        all_u = []
        for t in range(start_t, end_t):
            artist_array = [
                axis_observations.imshow(
                    make_uint8(axis_observations_image),
                    interpolation="none",
                    animated=True)
            ]

            # convert x,y into radians??
            # try reversing the camera direction calculation in rotate query viewpoint (impossible to reverse the linalg norm...)
            
            horizontal_angle_rad = np.arctan2(obs_viewpoint[0],obs_viewpoint[2]) 
            vertical_angle_rad = np.arcsin(obs_viewpoint[1]/camera_distance)
            
            # xz_diagonal = np.sqrt(np.square(obs_viewpoint[0])+np.square(obs_viewpoint[2]))
            
            # vertical_angle_rad = np.arctan2(obs_viewpoint[1],xz_diagonal)
            # vertical_angle_rad = np.arcsin(obs_viewpoint[1]/camera_distance)

            # horizontal_angle_rad, vertical_angle_rad = 0,0
            # ipdb.set_trace()
            horizontal_angle_rad, vertical_angle_rad = compute_vertical_rotation_at_frame(horizontal_angle_rad,vertical_angle_rad,
                t)
            if rotate_camera == False:
                horizontal_angle_rad, vertical_angle_rad = compute_camera_angle_at_frame(
                    0)
            
            query_viewpoints = rotate_query_viewpoint(
                horizontal_angle_rad, vertical_angle_rad, camera_distance)
            
            # obtain generated images, as well as mean and variance before gaussian
            generated_images, var_bg, latent_z, ct = model.generate_multi_image(query_viewpoints, representation, 100)
            logging.info("retrieved variables, time elapsed: "+str(time.time()-start_time))

            # cpu_generated_images = chainer.backends.cuda.to_cpu(generated_images)
            generated_images = np.squeeze(generated_images)

            latent_z =  np.squeeze(latent_z)
            # ipdb.set_trace()
            ct = np.squeeze(ct)
            
            # ht = np.squeeze(np.asarray(ht))
            # ut = np.squeeze(np.asarray(ut))
            
            # obtain data from Chainer Variable and obtain mean
            var_bg = cp.mean(var_bg,axis=0)
            logging.info("variance of bg, time elapsed: "+str(time.time()-start_time))
            var_z = cp.var(latent_z,axis=0)
            logging.info("variance of z, time elapsed: "+str(time.time()-start_time))
            # ipdb.set_trace()
            # print(ct.shape())
            var_c = cp.var(ct,axis=0)
            
            logging.info("variance of c, time elapsed: "+str(time.time()-start_time))
            # var_h = cp.var(ht,axis=0)
            # var_u = cp.var(ut,axis=0)
            
            # write viewpoint and image variance to file
            gen_img_var = np.var(generated_images,axis=0)
            logging.info("calculated variance of gen images, time elapsed: "+str(time.time()-start_time))

            all_var_bg.append((var_bg)[None])
            all_var.append((gen_img_var)[None])
            all_var_z.append((var_z)[None])
            all_q_view.append(chainer.backends.cuda.to_cpu(horizontal_angle_rad)[None]*180/math.pi)

            all_c.append((var_c)[None])
            logging.info("appending, time elapsed: "+str(time.time()-start_time))
            # all_h.append(chainer.backends.cuda.to_cpu(var_h)[None])
            # all_u.append(chainer.backends.cuda.to_cpu(var_u)[None])

            # sample = generated_images[0]
            pred_mean = cp.mean(generated_images,axis=0)
            
            # artist_array.append(
            #     axis_generation.imshow(
            #         make_uint8(pred_mean),
            #         interpolation="none",
            #         animated=True))            

            # animation_frame_array.append(artist_array)
        
        all_var_bg=np.concatenate(chainer.backends.cuda.to_cpu(all_var_bg),axis=0)
        all_var=np.concatenate(chainer.backends.cuda.to_cpu(all_var),axis=0)
        all_var_z = np.concatenate(chainer.backends.cuda.to_cpu(all_var_z),axis=0)

        all_c = np.concatenate(chainer.backends.cuda.to_cpu(all_c),axis=0)
        # all_h = np.concatenate(all_h,axis=0)
        # all_u = np.concatenate(all_u,axis=0)
        logging.info("concatenating, time elapsed: "+str(time.time()-start_time))

        with h5py.File(savename, "a") as f:
            f.create_dataset("variance_all_viewpoints",data=all_var)
            f.create_dataset("query_viewpoints",data=np.squeeze(np.asarray(all_q_view)))
            f.create_dataset("variance_b4_gaussian",data=all_var_bg)
            f.create_dataset("variance_of_z",data=all_var_z)

            f.create_dataset("c",data=all_c)
            # f.create_dataset("h",data=all_h)
            # f.create_dataset("u",data=all_u)
        logging.info("saving, time elapsed: "+str(time.time()-start_time))

    #==============================================================================
    # Visualization
    #==============================================================================
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(6, 7))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # fig.suptitle("GQN")
    axis_observations = fig.add_subplot(2, 1, 1)
    axis_observations.axis("off")
    axis_observations.set_title("observations")
    axis_generation = fig.add_subplot(2, 1, 2)
    axis_generation.axis("off")
    axis_generation.set_title("neural rendering")

    #==============================================================================
    # Generating animation
    #==============================================================================
    file_number = 1
    random.seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                animation_frame_array = []

                # shape: (batch, views, height, width, channels)
                images, viewpoints = subset[data_indices]
                camera_distance = np.mean(
                    np.linalg.norm(viewpoints[:, :, :3], axis=2))

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = preprocess_images(images)
                logging.info('preprocess '+str(time.time()-start_time))

                batch_index = 0

                total_views = images.shape[1]
                random_observation_view_indices = list(range(total_views))
                random.shuffle(random_observation_view_indices)
                random_observation_view_indices = random_observation_view_indices[:
                                                                                  total_observations_per_scene]

                #------------------------------------------------------------------------------
                # Observations
                #------------------------------------------------------------------------------
                observed_images = images[batch_index,
                                         random_observation_view_indices]
                observed_viewpoints = viewpoints[
                    batch_index, random_observation_view_indices]

                observed_images = to_device(observed_images)
                observed_viewpoints = to_device(observed_viewpoints)

                #------------------------------------------------------------------------------
                # Generate images with a single observation
                #------------------------------------------------------------------------------
                # Scene encoder
                representation = model.compute_observation_representation(
                    observed_images[None, :1], observed_viewpoints[None, :1])

                # Update figure
                observation_index = random_observation_view_indices[0]
                observed_image = images[batch_index, observation_index]
                axis_observations_image = fill_observations_axis(
                    [observed_image])

                # save observed viewpoint
                filename = "{}/variance_{}.hdf5".format(args.figure_directory, file_number)
                if os.path.exists(filename):
                    os.remove(filename)
                with h5py.File(filename, "a") as f:
                    f.create_dataset("observed_viewpoint",data=chainer.backends.cuda.to_cpu(observed_viewpoints[0]))
                    f.create_dataset("obs_viewpoint_horizontal_angle",data=np.arcsin(chainer.backends.cuda.to_cpu(observed_viewpoints[0][0])/camera_distance)*180/math.pi)

                logging.info('write 2 variables to hdf5 file, time elapsed: '+str(time.time()-start_time))
                obs_viewpoint = np.squeeze(observed_viewpoints[0])    
                # Neural rendering
                render(representation, camera_distance, observed_viewpoints[0], fps, fps * 6,
                       animation_frame_array, savename=filename)
                logging.info('write 4 other variables to hdf5 file, time elapsed: '+str(time.time()-start_time))
                #------------------------------------------------------------------------------
                # Write to file
                #------------------------------------------------------------------------------
                # anim = animation.ArtistAnimation(
                #     fig,
                #     animation_frame_array,
                #     interval=1 / fps,
                #     blit=True,
                #     repeat_delay=0)

                # anim.save(
                #     "{}/shepard_metzler_observations_{}.gif".format(
                #         args.figure_directory, file_number),
                #     writer="imagemagick",
                #     fps=fps)
                # anim.save(
                #     "{}/shepard_metzler_observations_{}.mp4".format(
                #         args.figure_directory, file_number),
                #     writer="ffmpeg",
                #     fps=2)
                
                if file_number==20:
                    break
                else:
                    file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--dataset-directory", type=str, required=True)
    parser.add_argument("--snapshot-directory", "-snapshot", type=str, required=True)
    parser.add_argument("--figure-directory", type=str, required=True)
    args = parser.parse_args()
    main()


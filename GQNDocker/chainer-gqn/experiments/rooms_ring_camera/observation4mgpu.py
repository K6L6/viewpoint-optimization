import argparse
import math
import time
import sys
import os
import random
import h5py
import ipdb
import cv2
from tqdm import tqdm
import chainer
import chainer.functions as cf
import cupy as cp
import numpy as np
from chainer.backends import cuda

# image processing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.filters import threshold_otsu

sys.path.append("./../../")
import gqn
from gqn.preprocessing import make_uint8, preprocess_images
from model_chain import Model
from hyperparams import HyperParameters
from functions import compute_yaw_and_pitch

def main():
    try:
        os.makedirs(args.figure_directory)
    except:
        pass


    #==============================================================================
    # Utilities
    #==============================================================================
    def read_files(directory):
        filenames = []
        files = os.listdir(directory)
        # ipdb.set_trace()
        for filename in files:
            if filename.endswith(".h5"):
                filenames.append(filename)
        filenames.sort()
        
        dataset_images = []
        dataset_viewpoints = []
        for i in range(len(filenames)):
            F = h5py.File(os.path.join(directory,filenames[i]))
            tmp_images = list(F["images"])
            tmp_viewpoints = list(F["viewpoints"])
            
            dataset_images.extend(tmp_images)
            dataset_viewpoints.extend(tmp_viewpoints)
    
        # for i in range(len(filenames)):
        #     images_npy_path = os.path.join(directory, "images", filenames[i])
        #     viewpoints_npy_path = os.path.join(directory, "viewpoints", filenames[i])
        #     tmp_images = np.load(images_npy_path)
        #     tmp_viewpoints = np.load(viewpoints_npy_path)
        
        #     assert tmp_images.shape[0] == tmp_viewpoints.shape[0]
            
        #     dataset_images.extend(tmp_images)
        #     dataset_viewpoints.extend(tmp_viewpoints)
        dataset_images = np.array(dataset_images)
        dataset_viewpoints = np.array(dataset_viewpoints)

        dataset = list()
        for i in range(len(dataset_images)):
            item = {'image':dataset_images[i],'viewpoint':dataset_viewpoints[i]}
            dataset.append(item)
        
        return dataset

    def to_device(array):
        # if using_gpu:
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
        return t * 2 * math.pi / (-fps * 2)

    def rotate_query_viewpoint(horizontal_angle_rad, camera_distance,
                               camera_position_y):
        camera_position = np.array([
            camera_distance * math.sin(horizontal_angle_rad),   # x
            camera_position_y,
            camera_distance * math.cos(horizontal_angle_rad),  # z
        ])
        center = np.array((0, camera_position_y, 0)) 
        camera_direction = camera_position - center
        yaw, pitch = compute_yaw_and_pitch(camera_direction)
         
        query_viewpoints = xp.array(
            (
                camera_position[0],
                camera_position[1],
                camera_position[2],
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
               camera_position_y,
               total_frames,
               animation_frame_array,
               rotate_camera=True):
        
        # viewpoint_file = open('viewpoints.txt','w')
        for t in range(0, total_frames):
            artist_array = [
                axis_observations.imshow(
                    cv2.cvtColor(make_uint8(axis_observations_image),cv2.COLOR_BGR2RGB),
                    interpolation="none",
                    animated=True)
            ]

            horizontal_angle_rad = compute_camera_angle_at_frame(t)
            if rotate_camera == False:
                horizontal_angle_rad = compute_camera_angle_at_frame(0)

            query_viewpoints = rotate_query_viewpoint(
                horizontal_angle_rad, camera_distance, camera_position_y)
            
            generated_images = model.generate_image(query_viewpoints,
                                                    representation)[0]
            generated_images = chainer.backends.cuda.to_cpu(generated_images)
            generated_images = make_uint8(generated_images)
            generated_images = cv2.cvtColor(generated_images, cv2.COLOR_BGR2RGB)

            artist_array.append(
                axis_generation.imshow(
                    generated_images,
                    interpolation="none",
                    animated=True))

            animation_frame_array.append(artist_array)

    def render_wVar(representation,
               camera_distance,
               camera_position_y,
               total_frames,
               animation_frame_array,
               no_of_samples,
               rotate_camera=True,
               wVariance=True):
        
        highest_var = 0.0
        with open("queries.txt",'w') as file_wviews, open("variance.txt",'w') as file_wvar:
            for t in range(0, total_frames):
                artist_array = [
                    axis_observations.imshow(
                        cv2.cvtColor(make_uint8(axis_observations_image),cv2.COLOR_BGR2RGB),
                        interpolation="none",
                        animated=True)
                ]

                horizontal_angle_rad = compute_camera_angle_at_frame(t)
                if rotate_camera == False:
                    horizontal_angle_rad = compute_camera_angle_at_frame(0)

                query_viewpoints = rotate_query_viewpoint(
                    horizontal_angle_rad, camera_distance, camera_position_y)
                
                q_x, q_y, q_z, _, _, _, _ = query_viewpoints[0]
                
                file_wviews.writelines("".join(str(q_x))+", "+
                                        "".join(str(q_y))+", "+
                                        "".join(str(q_z))+"\n")

                generated_images = cp.squeeze(cp.array(model.generate_images(query_viewpoints,
                                                        representation,no_of_samples)))
                # ipdb.set_trace()
                var_image = cp.var(generated_images,axis=0)
                mean_image = cp.mean(generated_images,axis=0)
                mean_image = make_uint8(np.squeeze(chainer.backends.cuda.to_cpu(mean_image)))
                mean_image_rgb = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)
                
                var_image = chainer.backends.cuda.to_cpu(var_image)

                # grayscale
                r,g,b = var_image
                gray_var_image = 0.2989*r+0.5870*g+0.1140*b            
                # thresholding Otsu's method
                # thresh = threshold_otsu(gray_var_image)
                # var_binary = gray_var_image > thresh

                # hill climb algorthm for searching highest variance
                cur_var = np.mean(gray_var_image)
                if cur_var>highest_var:
                    highest_var = cur_var

                    if wVariance==True:
                        print('highest variance: '+str(highest_var)+', viewpoint: '+str(query_viewpoints[0]))
                        highest_var_vp = query_viewpoints[0]
                        file_wvar.writelines('highest variance: '+str(highest_var)+', viewpoint: '+str(highest_var_vp)+'\n')
                    else:
                        pass

                artist_array.append(
                    axis_generation_var.imshow(
                        gray_var_image,
                        cmap=plt.cm.gray,
                        interpolation="none",
                        animated=True))

                artist_array.append(
                    axis_generation_mean.imshow(
                        mean_image_rgb,
                        interpolation="none",
                        animated=True))

                animation_frame_array.append(artist_array)

            if wVariance==True:
                print('final highest variance: '+str(highest_var)+', viewpoint: '+str(highest_var_vp))
                file_wvar.writelines('final highest variance: '+str(highest_var)+', viewpoint: '+str(highest_var_vp)+'\n')
            else:
                pass

        file_wviews.close()
        file_wvar.close()
    
    # loading dataset & model
    cuda.get_device(args.gpu_device).use()
    xp=cp

    hyperparams = HyperParameters()
    assert hyperparams.load(args.snapshot_directory)

    model = Model(hyperparams)
    chainer.serializers.load_hdf5(args.snapshot_file, model)
    model.to_gpu()

    total_observations_per_scene = 4
    fps = 30

    black_color = -0.5
    image_shape = (3, ) + hyperparams.image_size
    axis_observations_image = np.zeros(
        (3, image_shape[1], total_observations_per_scene * image_shape[2]),
        dtype=np.float32)

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
    axis_generation.set_title("Rendered Predictions")
    axis_generation_var = fig.add_subplot(2, 2, 3)
    axis_generation_var.axis("off")
    axis_generation_var.set_title("Variance Render")
    axis_generation_mean = fig.add_subplot(2, 2, 4)
    axis_generation_mean.axis("off")
    axis_generation_mean.set_title("Mean Render")
    
    
    # iterator
    dataset = read_files(args.dataset_directory)
    file_number = 1
    with chainer.no_backprop_mode():
        
        iterator  = chainer.iterators.SerialIterator(dataset,batch_size=1)
        # ipdb.set_trace()
        for i in tqdm(range(len(iterator.dataset))):
            animation_frame_array = []
            images, viewpoints = np.array([iterator.dataset[i]["image"]]),np.array([iterator.dataset[i]["viewpoint"]])
            
            camera_distance = np.mean(np.linalg.norm(viewpoints[:,:,:3],axis=2))
            camera_position_y = np.mean(viewpoints[:,:,1])

            images = images.transpose((0,1,4,2,3)).astype(np.float32)
            images = preprocess_images(images)

            batch_index = 0

            total_views = images.shape[1]
            random_observation_view_indices = list(range(total_views))
            random.shuffle(random_observation_view_indices)
            random_observation_view_indices = random_observation_view_indices[:total_observations_per_scene]
            observed_images = images[batch_index,
                                    random_observation_view_indices]
            observed_viewpoints = viewpoints[batch_index, 
                                            random_observation_view_indices]

            observed_images = to_device(observed_images)
            observed_viewpoints = to_device(observed_viewpoints)

            # Scene encoder
            representation = model.compute_observation_representation(
                observed_images[None, :1], observed_viewpoints[None, :1])

            # Update figure
            observation_index = random_observation_view_indices[0]
            observed_image = images[batch_index, observation_index]
            axis_observations_image = fill_observations_axis(
                [observed_image])

            # Neural rendering
            # render(representation, camera_distance, camera_position_y,
            #         fps * 2, animation_frame_array)
            render_wVar(representation, camera_distance, camera_position_y,
                    fps * 2, animation_frame_array, 100)
            
            for n in range(total_observations_per_scene):
                observation_indices = random_observation_view_indices[:n +
                                                                        1]
                axis_observations_image = fill_observations_axis(
                    images[batch_index, observation_indices])

                # Scene encoder
                representation = model.compute_observation_representation(
                    observed_images[None, :n + 1],
                    observed_viewpoints[None, :n + 1])
                # Neural rendering
                # render(representation, camera_distance, camera_position_y,
                #     fps // 2, animation_frame_array,rotate_camera=False)
                render_wVar(
                    representation,
                    camera_distance,
                    camera_position_y,
                    fps // 2,
                    animation_frame_array,
                    100,
                    rotate_camera=False,
                    wVariance=False)

            
            # Scene encoder with all given observations
            representation = model.compute_observation_representation(
                observed_images[None, :total_observations_per_scene + 1],
                observed_viewpoints[None, :total_observations_per_scene +
                                    1])

            # Neural rendering
            # render(representation, camera_distance, camera_position_y,
            #         fps * 6, animation_frame_array)
            render_wVar(representation, camera_distance, camera_position_y,
                    fps * 6, animation_frame_array,100)
            
            anim = animation.ArtistAnimation(
                        fig,
                        animation_frame_array,
                        interval=1/fps, # originally 1/fps
                        blit=True,
                        repeat_delay=0)

            anim.save(
                "{}/observations_{}.gif".format(
                    args.figure_directory, file_number),
                writer="imagemagick",
                fps=10)
            # ipdb.set_trace()
            # anim.save(
            #     "{}/rooms_ring_camera_observations_{}.mp4".format(
            #         args.figure_directory, file_number),
            #     writer='ffmpeg',
            #     fps=10)
            
            file_number += 1

# write the code to choose and output next viewpoint 
# better to keep the variance map in grayscale instead of using Otsu's threshold
# write the code to move the task arm to approach the object from a suitable grasping point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-directory", type=str, required=True)
    parser.add_argument("--snapshot-file",type=str,required=True)
    parser.add_argument("--dataset-directory", type=str, required=True)
    parser.add_argument("--figure-directory",type=str, required=True)
    parser.add_argument("--gpu-device",type=int,default=0)
    args = parser.parse_args()
    main()
    

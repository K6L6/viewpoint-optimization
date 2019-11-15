#from https://github.com/musyoku/gqn-dataset-renderer/blob/master/opengl/check.py
import os
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import ipdb

from PIL import Image


def main():
    action = True
    fig = plt.figure(figsize=(10, 10))
    image_filename_array = os.listdir(
        os.path.join(args.dataset_directory, "images"))
    # plt.title("number of objects:",len(image_array))
    # ipdb.set_trace()
    even = np.arange(2,12,2)
    c=0
    n=0
    m=100

    while action:
        for filename in image_filename_array:
            image_array = np.load(
                os.path.join(args.dataset_directory, "images", filename))
            # indices = np.random.choice(
            #     np.arange(image_array.shape[0]), replace=False, size=10 * 10)
            # ipdb.set_trace()
            # if (m >= image_array.shape[0]):
            #     break
            # else:
            indices = np.array([x for x in range(n,m)])
            images = image_array[indices]
            images = images[:, 0, ...]
            images = images.reshape((10, 10, 64, 64, 3))
            images = images.transpose((0, 2, 1, 3, 4))
            images = images.reshape((10 * 64, 10 * 64, 3))

            plt.imshow(images, interpolation="none")
            plt.pause(5)
            
            c+=1
            if c in even:
                n+=100
                m+=100
            else:
                continue
            
            # ans = input("continue? (y/n):")
            # if ans == 'y':
            #     n+=100
            #     m+=100
            #     continue
            # elif ans not in ('y','n'):
            #     print('invalid input')
            #     break
            # else:
            #     action=False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-directory", "-dataset", type=str, required=True)
    args = parser.parse_args()
    main()
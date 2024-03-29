import os
import numpy as np


class Subset():
    '''modified to compare with ground truth'''
    def __init__(self, image_npy_path, viewpoints_npy_path, original_images_npy_path=""):
        self.images = np.load(image_npy_path)
        self.viewpoints = np.load(viewpoints_npy_path)
        self.use_ground_truth = not original_images_npy_path == ""
        if self.use_ground_truth:
            self.original_images = np.load(original_images_npy_path)
        assert self.images.shape[0] == self.viewpoints.shape[0]

    def __getitem__(self, indices):
        if self.use_ground_truth:
            return self.images[indices], self.viewpoints[indices], self.original_images[indices]
        else:
            return self.images[indices], self.viewpoints[indices]

    def __len__(self):
        return self.images.shape[0]

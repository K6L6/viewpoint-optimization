import argparse
import json
import os
import pandas as pd
import time

import chainer
import chainer.functions as cf
import cupy as cp
import numpy as np
from chainer.backends import cuda

def log_loss(self,directory,epochs,MSE,elapsed_time):
    self.start_time = time.time()
    self.log = pd.data(columns=["epoch","train:MSE","elapsed_time(min)",])
    self.snapshot_filename = "loss.csv"
    
    data=[epochs]
    data.append(MSE)
    data.append(elapsed_time)

    elapsed_time = (time.time()-self.start_time)/60
    csv_path = os.path.join(directory,self.snapshot_filename)
    self.log.to_csv(csv_path, index=False)

def read_dataset(self,directory):
    self.directory = directory
    self.file_list = []
    files = os.listdir(os.path.join(self.directory,"images"))
    for filename in files:
        if filename.endswith(".npy"):
            self.file_list.append(filename)
    self.file_list.sort()

    img_path = os.path.join(self.directory, "images", filename)
    uncertainty_path = os.path.join(self.directory, "uncertainties", filename)

    self.images = np.load(img_path)
    self.uncertainties = np.load(uncertainty_path)
    assert self.images.shape[0] == self.uncertainties[0]

    return self.images, self.uncertainties

def sampler(self,data,batch_size):
    self.data=data
    self.batch_size=batch_size
    batch = []
    for entry in data:
        batch.append(int(entry))
        if len(batch)==self.batch_size:
            yield batch
            batch=[]

def save(self,directory):
    self.epochs = epochs
    self.batch = batch_size
    self.elapsed_time = elapsed_time
    self.snapshot_filename = "model.json"
    path = os.path.join(directory, self.snapshot_filename)
    with open(path,"w") as f:
        json.dump({
            "epoch":self.epochs,
            "batch size":self.batch,
            "elapsed time":self.elapsed_time,
        },f)

def main():
    os.makedirs(args.log_directory)
    os.makedirs(args.model_directory)

    # GPU usage
    gp = np
    gpu_device = args.gpu_device
    assigned_gpu = gpu_device>=0
    if assigned_gpu:
        cuda.get_device(gpu_device).use()
        gp = cp

    # Dataset
    data_train = read_dataset(args.train_dataset_directory)
    if args.test_dataset_directory is not None:
        data_test = read_dataset(args.test_dataset_directory)
    
    # # Logging
    # csv = data_format
    # csv.load(args.logdirectory)

    # Optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001, 
                                        beta1=0.9, 
                                        beta2=0.999, 
                                        eps=1e-08, 
                                        eta=1.0, 
                                        weight_decay_rate=0, 
                                        amsgrad=False, 
                                        adabound=False, 
                                        final_lr=0.1, 
                                        gamma=0.001)
    print(optimizer)    

    # Training
    dataset_size = len(data_train)

    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))

        for subset_index,subset in enumerate(data_train):
            batches = sampler(data, batch_size=args.batch_size)

            for batch_index, data in enumerate(batches):

                images, uncertainties = subset[data]

                z = 
                 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-directory","-train", type=str, required=True)
    parser.add_argument("--test-dataset-directory","-test", type=str, default=None)
    parser.add_argument("--gpu-device","-gpu",type=int,default=0)
    parser.add_argument("--batch-size","-batch", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-directory","-log", type=str,default="./log")
    parser.add_argument("--model-directory","-model",type=str,default="./model")

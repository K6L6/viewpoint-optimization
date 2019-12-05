# Notes on GQN Docker
chainer-gqn folder is a repo cloned from (https://github.com/musyoku/chainer-gqn.git), and also contains modified scripts to run training over multiple GPUs.  
  
copydata.sh is just a script that's used to copy a subset of files from a large dataset folder.  
  
Dockerfile has all the chainer-gqn dependencies in a conda environment.  

## To build Dockerfile
```docker build -t [name of docker image] .```

## To link files with the local PC, when running the container
```nvidia-docker run -it --rm -v $PWD/[foldername]:[path/in/container] [image_name]```

## Below is just a bunch of text, related to training done within the GQN container or just execution commands for copy pasting
GQN paper shows data of results up to 2 million iterations. Stated in the supplementary, section 5.6

## data format example
shape of image data is [2000, 15, 64, 64, 3]  
scenes = 2000  
images per object = 15  
pixels 64 x 64  
3 channels RGB  
  
shape of viewpoint data is [2000, 15, 7]  

## notes on CNN
[zero_padding = (K-1)/2] where K is filter size  
[conv_out_layer height/length = ((W-K+2P)/S)+1] where  
W is input height and length,  
K is filter size,  
P is padding,  
S is stride  

## GQN representation network
inp = 64x64 = 4096  
conv1_1 = inp_channels:3, out_channels:256,   filter_size=2 (2x2), pad=0, stride=2  

### current docker run commands
```nvidia-docker run -it --name gqn-train --rm -v $PWD/gqn-datasets:/GQN/gqn-datasets -v $PWD/chainer-gqn:/GQN/chainer-gqn kelvin/gqn```

```nvidia-docker run -it --name kelvin2 --rm -v $PWD/gqn-datasets:/GQN/gqn-datasets -v $PWD/chainer-gqn:/GQN/chainer-gqn kelvin/gqn```

```nvidia-docker run -it --name kelvin3 --rm -v $PWD/dataset_shepard_matzler_train:/GQN/gqn-datasets -v $PWD/chainer-gqn:/GQN/chainer-gqn kelvin/gqn```

```nvidia-docker run -it --name kelvin3 --rm -v /media/HDPH-UT/2018-GQN/datasets/m_train_o_d1_k15:/GQN/gqn-datasets -v /home/kelvin/OgataLab/latest-chainer-gqn:/GQN/chainer-gqn kelvin/gqn```

```nvidia-docker run -it --name sim_dataset_train --rm -v ~/GQNDocker/gqn-datasets/dataset4GQN:/GQN/gqn-datasets/train -v ~/GQNDocker/chainer-gqn:/GQN/chainer-gqn kelvin/gqn```

### data translator container
```nvidia-docker run -it --name dat-trans --rm -v $PWD/gqn-datasets:/GQN/gqn-datasets -v $PWD/chainer-gqn:/GQN/chainer-gqn -v $PWD/gqn-datasets-translator:/GQN/dataset-translator kelvin/gqn```

needs torch  

### GQN dataset download and convert
```gsutil rsync -r gs://gqn-dataset/shepard_metzler_7_parts /GQN/gqn-datasets/shepard_metzler_7_parts```

```gsutil rsync -r gs://gqn-dataset/rooms_free_camera_with_object_rotations /GQN/gqn-datasets/rooms_free_camera_with_object_rotations```

```python3 convert.py --dataset-name shepard_metzler_7_parts --working-directory /GQN/gqn-datasets/ --dataset-directory /GQN/gqn-datasets/shepard_metzler_7_parts```

```python3 merge.py --source-directory /GQN/gqn-datasets/ --output-directory /GQN/gqn-datasets/shepard_metzler_7_npy```

### dataset
shepard_metzler_5_npy=[train:scenes=809000,images=809000*15,140.419GB],[test:scenes=199000,images=199000*15,35.103GB]
2018-data[m_train_o_d1_k15]=[train:scenes=18000,images=18000*15,3.1081GB],[test:scenes=2000,images=2000*15,353.914MB]
2018-data[m_train_d100_k15]=[scenes=2000000,images=2000000*15,345.893GB]
### Training
test batch size for evaluation is set to batchsize*6 as default.

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/train0.25 --test-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/test0.25 --snapshot-directory ./snapshots/1_sm5_d0.25_b20_e1000_gs12 --log-directory ./log/1_sm5_d0.25_b20_e1000_gs12 --batch-size 20 --gpu-device 0 --visualize```

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/train0.1 --test-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/test0.1 --snapshot-directory ./snapshots/1_sm5_d0.1_b20_e1000_gs12 --log-directory ./log/1_sm5_d0.1_b20_e1000_gs12 --batch-size 20 --gpu-device 1 --visualize```

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/train --test-dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/test --snapshot-directory ./snapshots/1_sm5_d1_b20_e1_gs12 --log-directory ./log/1_sm5_d1_b20_e1_gs12 --batch-size 20 --epochs 1 --gpu-device 0```	(592.088 min)

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/train --test-dataset-directory /GQN/gqn-datasets/test --snapshot-directory ./snapshots/1_sm2018_d1_b20_e1_gs12 --log-directory ./log/1_sm2018_d1_b20_e1_gs12 --batch-size 20 --epochs 1 --gpu-device 1```

data size = 1.4T  
Test:  
        ELBO: -14979.2060546875 - bits/pixel: 9.758660316467285 - MSE: 8.585481749534552e-07 - done in 318.187 min  
Epoch 1 done in 1549.942 min  
    ELBO: -17552.40134202881 - bits/pixel: 10.060771656358241 - MSE: 3.3823487039936496e-05
    lr: 9.550005625000001e-05 - sigma: 1.35 - training_steps: 80000
    Time elapsed: 1549.969 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --test-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/test --snapshot-directory ./snapshots/2_sm2018_d1_b20_e1_gs12 --log-directory ./log/2_sm2018_d1_b20_e1_gs12 --batch-size 20 --epochs 1 --gpu-device 0```

Test:  
        ELBO: -19794.284790039062 - bits/pixel: 10.323984146118164 - MSE: 0.01823753141798079 - done in 0.424 min  
Epoch 1 done in 12.249 min  
    ELBO: -19840.519991319445 - bits/pixel: 10.329412435955472 - MSE: 0.024451180640608073
    lr: 9.994943125e-05 - sigma: 1.9926875 - training_steps: 900
    Time elapsed: 12.274 min

#### test batchsize = batch size
```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --test-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/test --snapshot-directory ./snapshots/3_sm2018_d1_b30_e100_gs12 --log-directory ./log/3_sm2018_d1_b30_e100_gs12 --batch-size 30 --epochs 100 --gpu-device 0```

Test:  
        ELBO: -16419.240589488636 - bits/pixel: 9.92773082039573 - MSE: 0.0011612985355454978 - done in 0.427 min  
Epoch 100 done in 11.961 min  
    ELBO: -16438.90973537458 - bits/pixel: 9.930040179679692 - MSE: 0.0012151322170569929
    lr: 9.665880625e-05 - sigma: 1.517375 - training_steps: 59400
    Time elapsed: 1195.469 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --test-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/test --snapshot-directory ./snapshots/4_sm2018_d0.01_b30_e200_gs12 --log-directory ./log/4_sm2018_d0.01_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 0```

Test:  
        ELBO: -11715.860765861742 - bits/pixel: 9.375521876595236 - MSE: 0.0006713226528407893 - done in 0.432 min  
Epoch 200 done in 12.016 min  
    ELBO: -11744.222500065762 - bits/pixel: 9.378851754095418 - MSE: 0.0006268925706915532
    lr: 9.331755625000001e-05 - sigma: 1.0347499999999998 - training_steps: 118800
    Time elapsed: 2428.995 min approx:40.5hrs

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/shepard_metzler_7_npy/train0.02 --test-dataset-directory /GQN/gqn-datasets/shepard_metzler_7_npy/test0.02 --snapshot-directory ./snapshots/1_sm7_d0.02_b30_e200_gs12 --log-directory ./log/1_sm7_d0.02_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 0```

Test:  
        ELBO: -11715.540571732954 - bits/pixel: 9.375484249808572 - MSE: 0.0005932653368676476 - done in 0.422 min  
Epoch 200 done in 11.821 min  
    ELBO: -11744.100959135627 - bits/pixel: 9.378837466641308 - MSE: 0.0005847357418007917
    lr: 9.331755625000001e-05 - sigma: 1.0347499999999998 - training_steps: 118800
    Time elapsed: 2397.500 min approx:40hrs

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_1 --test-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/test_1 --snapshot-directory ./snapshots/1_roomsFCwOR_d0.0018_b30_e200_gs12 --log-directory ./log/1_roomsFCwOR_d0.0018_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 1```

Test:  
        ELBO: -11714.84296579072 - bits/pixel: 9.375402363863858 - MSE: 0.0004774779835398394 - done in 0.409 min  
Epoch 200 done in 11.569 min  
    ELBO: -11743.502494015678 - bits/pixel: 9.378767206211283 - MSE: 0.0004770516411355079
    lr: 9.331755625000001e-05 - sigma: 1.0347499999999998 - training_steps: 118800
    Time elapsed: 2310.353 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --test-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/test --snapshot-directory ./snapshots/4_sm2018_d0.01_b30_e200_gs12 --log-directory ./log/4_sm2018_d0.01_b30_e200_gs12 --batch-size 30 --epochs 400 --gpu-device 0```

Test:  
        ELBO: -6914.091397372159 - bits/pixel: 8.81176135034272 - MSE: 0.0003696193091331445 - done in 0.428 min  
Epoch 400 done in 11.996 min  
    ELBO: -6913.521444095907 - bits/pixel: 8.811694525709056 - MSE: 0.00032714780127360743
    lr: 8.663505625e-05 - sigma: 0.7 - training_steps: 237600
    Time elapsed: 2424.460 min + 2428.995 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --snapshot-directory ./snapshots/4_sm2018_d0.01_b30_e200_gs12 --log-directory ./log/4_sm2018_d0.01_b30_e200_gs12 --batch-size 30 --epochs 600 --gpu-device 1```

Epoch 600 done in 11.333 min  
    ELBO: -6912.320146451494 - bits/pixel: 8.811553492690578 - MSE: 0.00023704027279432212
    lr: 7.995255625e-05 - sigma: 0.7 - training_steps: 356400
    Time elapsed: 2290.954 min + 2424.460 min + 2428.995 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --snapshot-directory ./snapshots/4_sm2018_d0.01_b30_e200_gs12 --log-directory ./log/4_sm2018_d0.01_b30_e200_gs12 --batch-size 30 --epochs 800 --gpu-device 1```

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/train --snapshot-directory ./snapshots/5_gazebo_d1_b30_e200_gs12 --log-directory ./log/5_gazebo_d1_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 0 --image-size 800``` #failed due to memory

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/train --snapshot-directory ./snapshots/5_gazebo_d1_b15_e200_gs12 --log-directory ./log/5_gazebo_d1_b15_e200_gs12 --batch-size 15 --epochs 200 --gpu-device 0```
Epoch 200 done in 0.511 min  
    ELBO: -19759.60107421875 - bits/pixel: 10.31991195678711 - MSE: 0.0154473666
33452475
    lr: 9.991005625e-05 - sigma: 1.9869999999999999 - training_steps: 1600
    Time elapsed: 106.462 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/train --snapshot-directory ./snapshots/5_gazebo_d1_b20_e200_gs12 --log-directory ./log/5_gazebo_d1_b20_e200_gs12 --batch-size 20 --epochs 200 --gpu-device 0```
Epoch 200 done in 0.483 min  
    ELBO: -19819.6953125 - bits/pixel: 10.326967716217041 - MSE: 0.019531857687979937
    lr: 9.995505625e-05 - sigma: 1.9935 - training_steps: 800
    Time elapsed: 100.832 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train/ --snapshot-directory ./snapshots/6_roomswrotation_d1_b30_e200_gs12 --log-directory ./log/6_roomswrotation_d1_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 1```
stopped at epoch 3 subset 601 over 12441.554 min elapsed  

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train2000/ --snapshot-directory ./snapshots/6_roomswrotation_d0.2_b30_e200_gs12 --log-directory ./log/6_roomswrotation_d0.2_b30_e200_gs12 --batch-size 30 --epochs 200 --gpu-device 2```
stopped at epoch 8 subset 801 over 10909min  

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_0.02/ --snapshot-directory ./snapshots/6_roomswrotation_d0.02_b25_e200_gs12 --log-directory ./log/6_roomswrotation_d0.02_b25_e200_gs12 --batch-size 25 --epochs 200 --gpu-device 3```
stopped at epoch 23 subset 101 over 133.831 min  

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_1/ --snapshot-directory ./snapshots/6_roomswrotation_d0.002_b30_e100_gs12 --log-directory ./log/6_roomswrotation_d0.002_b30_e100_gs12 --batch-size 30 --epochs 100 --gpu-device 0```
Epoch 100 done in 11.155 min  
    ELBO: -16437.906391387838 - bits/pixel: 9.92992236236932 - MSE: 0.0008181034452975681
    lr: 9.665880625e-05 - sigma: 1.517375 - training_steps: 59400
    Time elapsed: 1111.863 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_20npy/ --snapshot-directory ./snapshots/6_roomswrotation_d20npy_b30_e300_gs12 --log-directory ./log/6_roomswrotation_d20npy_b30_e300_gs12 --batch-size 30 --epochs 300 --gpu-device 0```
Epoch 300 done in 12.373 min  
    ELBO: -6915.624245383523 - bits/pixel: 8.811941380934282 - MSE: 0.00046428280047345623
    lr: 8.886255625e-05 - sigma: 0.7 - training_steps: 198000
    Time elapsed: 3704.746 min

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_20npy/ --snapshot-directory ./snapshots/6_roomswrotation_d20npy_b30_e300_gs12 --log-directory ./log/6_roomswrotation_d20npy_b30_e300_gs12 --batch-size 30 --epochs 1000 --gpu-device 0```
Epoch 1000 done in 12.895 min  
    ELBO: -6912.9973477450285 - bits/pixel: 8.811632975665006 - MSE: 0.0002769026601251398
    lr: 6.287505625e-05 - sigma: 0.7 - training_steps: 660000
    Time elapsed: 8878.137 min continued from 300

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/sim_dataset8pf/train/ --test-dataset-directory /GQN/gqn-datasets/sim_dataset8pf/validation/ --snapshot-directory ./snapshots/5_gazebo_d128_b30_e1000_gs12 --log-directory ./log/5_gazebo_d128_b30_e1000_gs12 --batch-size 30 --epochs 1000 --gpu-device 0```
Epoch 1000 done in 0.508 min  
    ELBO: -19098.303571428572 - bits/pixel: 10.242271287100655 - MSE: 0.00361004
36451046596
    lr: 9.921255625000001e-05 - sigma: 1.88625 - training_steps: 14000
    Time elapsed: 553.409 min

```python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_20npy/ --test-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/test_1 --snapshot-directory ./snapshots/6_roomswrotation_d20npy_b30_e300_gs12test --log-directory ./log/6_roomswrotation_d20npy_b30_e300_gs12test --batch-size 30 --epochs 300 --ngpu 1```

```python3 train.py --train-dataset-directory /GQN/gqn-datasets/sim_dataset8pf/train/ --test-dataset-directory /GQN/gqn-datasets/sim_dataset8pf/validation/ --snapshot-directory ./snapshots/5_gazebo_d128_b5_e1000_gs12 --log-directory ./log/5_gazebo_d128_b5_e1000_gs12 --batch-size 5 --epochs 1500 --gpu-device 0```
Epoch 1500 done in 0.703 min  
    ELBO: -18719.797712053572 - bits/pixel: 10.19783183506557 - MSE: 0.0025425389342542204
    lr: 9.881880625e-05 - sigma: 1.829375 - training_steps: 21000
    Time elapsed: 446.405 min

## Multi-GPU test
```python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_20npy/ --test-dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/test_1/ --snapshot-directory ./snapshots/6_roomswrotation_d20npy_b30_e300_gs12test --log-directory ./log/6_roomswrotation_d20npy_b30_e300_gs12test --batch-size 30 --epochs 300 --ngpu 1```
approx 6 days, due to saving the model every epoch.

```python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/simdataset8pf128x128/train/ --test-dataset-directory /GQN/gqn-datasets/simdataset8pf128x128/validation/ --snapshot-directory ./snapshots/5_gazebomgpu_d128_b5_e2000_im128_gs12test --log-directory ./log/5_gazebomgpu_d128_b5_e2000_im128_gs12test --image-size 128 --batch-size 5 --epochs 2000 --ngpu 2```

52547.9 seconds (14hours 35min)  

```python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64/train/ --test-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64/validation/ --snapshot-directory ./snapshots/5_gazebo_d128_b5_e2000_im64_gs12 --log-directory ./log/5_gazebo_d128_b5_e2000_im64_gs12 --image-size 64 --batch-size 5 --epochs 2000 --ngpu 2```

"main/loss": 19813.013227982956,  
"main/bits_per_pixel": 10.326183102347635,  
"main/NLL": 19812.54607599432,  
"main/MSE": 0.00210533589548008,  
"validation/main/ELBO": -19814.189453125,  
"validation/main/bits_per_pixel": 10.326321601867676,  
"validation/main/NLL": 19813.501953125,  
"validation/main/KLD": 0.6883742809295654,  
"validation/main/MSE": 0.0027262240182608366,  
"epoch": 2000,  
"iteration": 44800,  
"elapsed_time": 26470.47290384583 (7hours 21min)   

```python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64/train/ --test-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64/validation/ --snapshot-directory ./snapshots/5_gazebo_d128_b5_e5000_im64_gs12 --log-directory ./log/5_gazebo_d128_b5_e5000_im64_gs12 --image-size 64 --batch-size 5 --epochs 5000 --ngpu 2```

"main/loss": 19810.5703125,  
"main/bits_per_pixel": 10.325896219773727,  
"main/NLL": 19810.417613636364,  
"main/MSE": 0.0007189067949763161,  
"validation/main/ELBO": -19817.0703125,  
"validation/main/bits_per_pixel": 10.326659202575684,  
"validation/main/NLL": 19816.861328125,  
"validation/main/KLD": 0.2081449329853058,  
"validation/main/MSE": 0.0049134804867208,  
"epoch": 5000,  
"iteration": 112000,  
"elapsed_time": 64022.753862868994 (17hours 47min)  

python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64bot_only/train/ --test-dataset-directory /GQN/gqn-datasets/simdataset8pf64x64bot_only/validation/ --snapshot-directory ./snapshots/5_gazebo_d128_b5_e5000_im64bot_gs12 --log-directory ./log/5_gazebo_d128_b5_e5000_im64bot_gs12 --image-size 64 --batch-size 5 --epochs 5000 --ngpu 2 --report-interval-iters 100  

62888.5s  

python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/train/ --test-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/validation/ --snapshot-directory ./snapshots/6_RFCNOR_d128_b5_e2500_im64_gs12/ --log-directory ./log/6_RFCNOR_d128_b5_e2500_im64_gs12 --image-size 64 --batch-size 5 --epochs 2500 --ngpu 2  

28643.9s  

python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/train/ --test-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/validation/ --snapshot-directory ./snapshots/6_RFCNOR_d128_b5_e5000_im64_gs12 --log-directory ./log/6_RFCNOR_d128_b5_e5000_im64_gs12 --image-size 64 --batch-size 5 --epochs 5000 --ngpu 2 --report-interval-iters 50  

58844.2s  

python3 train_mGPU.py --train-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/train/ --test-dataset-directory /GQN/gqn-datasets/RFCNOR_subset/validation/ --snapshot-directory ./snapshots/6_RFCNOR_d128_b5_e10000_im64_gs12 --log-directory ./log/6_RFCNOR_d128_b5_e10000_im64_gs12 --image-size 64 --batch-size 5 --epochs 10000 --ngpu 2 --report-interval-iters 100  
stopped midway because data was incompatible  

### data list creation
```ls ./gqn-datasets/shepard_metzler_5_npy/test/images/*.npy | head -n 50 > testi.list```
```less testi.list``` 
```vim copydata.sh```
```./copydata.sh```

# Observation in docker
```python3 observation.py --dataset-directory /GQN/gqn-datasets/shepard_metzler_5_npy/train --snapshot-directory /GQN/chainer-gqn/snapshots/1_sm5_d1_b20_e1_gs12 --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/1_sm5_d1_b20_e1_gs12 --camera-distance 100```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/ --snapshot directory /GQN/chainer-gqn/snapshots/1_sm2018_d1_b20_e1_gs12 --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/1_sm2018_d1_b20_e1_gs12 --camera-distance 100```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train --snapshot-directory /GQN/chainer-gqn/snapshots/2_sm2018_d1_b20_e1_gs12 --gpu-device 1 --figure-directory /GQN/chainer-gqn/figures/2_sm2018_d1_b20_e1_gs12 --camera-distance 100```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/3_sm2018_d1_b30_e100_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/3_sm2018_d0.01_b30_e100_gs12_c100 --camera-distance 100```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/5_gazebo_d1_b20_e200_gs12/ --figure-directory /GQN/chainer-gqn/figures/5_gazebo_d1_b20_e200_gs12 --gpu-device 0```

## camera distance parameter was removed from observation.py on 9th May.

```python3 observation.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/3_sm2018_d1_b30_e100_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/3_sm2018_d0.01_b30_e100_gs12```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/2_sm2018_d0.01_b20_e1_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/3_sm2018_d0.01_b20_e1_gs12```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/4_sm2018_d0.01_b30_e200_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/4_sm2018_d0.01_b30_e200_gs12```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/shepard_metzler_7_npy/train0.02/ --snapshot-directory /GQN/chainer-gqn/snapshots/1_sm7_d0.02_b30_e200_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/1_sm7_d0.02_b30_e200_gs12```

```python3 observation.py --dataset-directory /GQN/gqn-datasets/rooms_free_camera_with_object_rotations_npy/train_1 --snapshot-directory /GQN/chainer-gqn/snapshots/1_roomsFCwOR_d0.0018_b30_e200_gs12 --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/1_roomsFCwOR_d0.0018_b30_e200_gs12```

```python3 observation_testing.py --dataset-directory /GQN/gqn-datasets/2018-data/m_train_o_d1_k15/train/ --snapshot-directory /GQN/chainer-gqn/snapshots/4_sm2018_d0.01_b30_e200_gs12/ --gpu-device 0 --figure-directory /GQN/chainer-gqn/figures/4_sm2018_d0.01_b30_e200_gs12_test```

```python3 horizontal_observation.py --dataset-directory ~/gqn-datasets/m_train_o_d1_k15/train --snapshot-directory ~/chainer-gqn/snapshots/4_sm2018_d0.01_b30_e400_gs12_copy/ --gpu-device 0 --figure-directory ~/chainer-gqn/figures/4_sm2018_d0.01_b30_e400_gs12_v0trained_tensXtest```

# notes on GQN code
intrinsic rotations  
yaw = z  
pitch = y'  
roll = x"  

# Difference between 2018 GQN and the current GQN

Old GQN layers:  
['0', ['lstm'], '1', ['lstm'], '10', ['lstm'], '11', ['lstm'], '12', ['conv'], '13', ['conv'], '14', ['conv'], '15', ['conv'], '16', ['conv'], '17', ['conv'], '18', ['conv'], '19', ['conv'], '2', ['lstm'], '20', ['conv'], '21', ['conv'], '22', ['conv'], '23', ['conv'], '24', ['deconv'], '25', ['deconv'], '26', ['deconv'], '27', ['deconv'], '28', ['deconv'], '29', ['deconv'], '3', ['lstm'], '30', ['deconv'], '31', ['deconv'], '32', ['deconv'], '33', ['deconv'], '34', ['deconv'], '35', ['deconv'], '36', ['W', 'b'], '37', ['lstm'], '38', ['lstm'], '39', ['lstm'], '4', ['lstm'], '40', ['lstm'], '41', ['lstm'], '42', ['lstm'], '43', ['lstm'], '44', ['lstm'], '45', ['lstm'], '46', ['lstm'], '47', ['lstm'], '48', ['lstm'], '49', ['conv'], '5', ['lstm'], '50', ['conv'], '51', ['conv'], '52', ['conv'], '53', ['conv'], '54', ['conv'], '55', ['conv'], '56', ['conv'], '57', ['conv'], '58', ['conv'], '59', ['conv'], '6', ['lstm'], '60', ['conv'], '61', ['conv_1', 'conv_2', 'conv_3'], '62', ['conv1_1', 'conv1_2', 'conv1_3', 'conv1_res', 'conv2_1', 'conv2_2', 'conv2_3', 'conv2_res'], '7', ['lstm'], '8', ['lstm'], '9', ['lstm']]

Current GQN layers:  
['/', '/0', '/0/broadcast_v', '/0/lstm', '/0/upsample_h', '/1', '/1/broadcast_v', '/1/lstm', '/1/upsample_h', '/2', '/2/broadcast_v', '/2/lstm', '/2/upsample_h', '/3', '/3/broadcast_v', '/3/lstm', '/3/upsample_h', '/4', '/4/broadcast_v', '/4/lstm', '/4/upsample_h', '/5', '/5/broadcast_v', '/5/lstm', '/5/upsample_h', '/6', '/6/broadcast_v', '/6/lstm', '/6/upsample_h', '/7', '/7/broadcast_v', '/7/lstm', '/7/upsample_h', '/8', '/8/broadcast_v', '/8/lstm', '/8/upsample_h', '/9', '/9/broadcast_v', '/9/lstm', '/9/upsample_h', '/10', '/10/broadcast_v', '/10/lstm', '/10/upsample_h', '/11', '/11/broadcast_v', '/11/lstm', '/11/upsample_h', '/12', '/12/conv', '/13', '/14', '/14/broadcast_v', '/14/downsample_xu', '/14/lstm', '/15', '/15/broadcast_v', '/15/downsample_xu', '/15/lstm', '/16', '/16/broadcast_v', '/16/downsample_xu', '/16/lstm', '/17', '/17/broadcast_v', '/17/downsample_xu', '/17/lstm', '/18', '/18/broadcast_v', '/18/downsample_xu', '/18/lstm', '/19', '/19/broadcast_v', '/19/downsample_xu', '/19/lstm', '/20', '/20/broadcast_v', '/20/downsample_xu', '/20/lstm', '/21', '/21/broadcast_v', '/21/downsample_xu', '/21/lstm', '/22', '/22/broadcast_v', '/22/downsample_xu', '/22/lstm', '/23', '/23/broadcast_v', '/23/downsample_xu', '/23/lstm', '/24', '/24/broadcast_v', '/24/downsample_xu', '/24/lstm', '/25', '/25/broadcast_v', '/25/downsample_xu', '/25/lstm', '/26', '/26/conv', '/27', '/27/broadcast_v', '/27/conv1_1', '/27/conv1_2', '/27/conv1_3', '/27/conv1_res', '/27/conv2_1', '/27/conv2_2', '/27/conv2_3', '/27/conv2_res']

MSE with chainer 0.02962886  
method = cf.mean_squared_error(obs_image,pred_image)  
  
MSE with flatten 0.02962885  
method = np.square(obs_image.flatten()-pred_image.flatten()).mean()  
  
MSE with flatten in my_MSE 0.02962784  
MSE with unpack_MSE 0.02962784  
  
Training the model for viewpoint optimization.  
use the same method as when observing for uncertainty with variance.  
sample 100 predictions from one representation and find the viewpoint observing highest uncertainty.  
Use the hill-climbing algorithm to search for the optimal viewpoint in the entire dome.  
discretize around 100 to 1000 viewpoints.  


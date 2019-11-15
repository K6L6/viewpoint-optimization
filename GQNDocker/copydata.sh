#!/bin/bash
# ls ${folder}/*.npy | head/tail -n ${number of lines} > filelist

list=$(cat train.list)
outfolder=./gqn-datasets/shepard_metzler_5_npy/train0.25/images
mkdir -p $outfolder
for i in ${list}; do
    cp $i $outfolder
done

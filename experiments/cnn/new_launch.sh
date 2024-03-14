#!/bin/bash                                          
n_shots=(-1)
bs=(128)
seeds=(2)
models=(deit_s swin_s)
datasets=(cifar10 oxfordpets dtd  gtsrb)
bits=(3 4)

for i in ${models[@]}
do
  for j in ${datasets[@]}
  do 
    for k in ${bits[@]} 
    do 
      bash new_setoff.sh --model $i --wandb --n_shot -1 --batch_size 128 --dataset $j --bits $k  --seed 2
    done
  done
done


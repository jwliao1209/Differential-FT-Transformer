#!/bin/bash

data_ids=(
    361068
    361274
    361276
    361113
)

for data_id in ${data_ids[@]}; do
    python run.py \
        --data_dir /home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data \
        --data_id $data_id \
        --model ftt \
        --n_epoch 300 \
        --target_transform

    python run.py \
        --data_dir /home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data \
        --data_id $data_id \
        --model dftt \
        --n_epoch 300 \
        --target_transform
done

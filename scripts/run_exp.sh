#!/bin/bash

data_ids=(
    361055
    361060
    361061
    361062
    361065
    361069
    361068
    361274
    361113
    361072
    361073
    361074
    361076
    361077
    361276
    361097
    361287
)

models=(
    ft
    diff
    dint
)

for data_id in ${data_ids[@]}; do
    for model in ${models[@]}; do
        python run.py \
            --data_dir /home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data \
            --data_id $data_id \
            --model $model \
            --n_epoch 300 \
            --target_transform
    done
done

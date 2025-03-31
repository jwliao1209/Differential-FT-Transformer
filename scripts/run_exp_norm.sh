#!/bin/bash

data_ids=(
    361055
    361060
    361061
    361062
    361065
    361069
    # 361068
    # 361274
    # 361113
    # 361072 #
    # 361073
    # 361074
    # 361076
    # 361077 #
    # 361276
    # 361097
    # 361287
)

models=(
    dofen
    # ft
    # diff
    # dint
)

norms=(
    # layer_norm
    # dyt
    # dyas
    dyat
    fdyat
)

for data_id in ${data_ids[@]}; do
    for model in ${models[@]}; do
        for norm in ${norms[@]}; do
            python run.py \
                --project_name DOFEN_norm \
                --data_dir /home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data \
                --data_id $data_id \
                --model $model \
                --norm $norm \
                --n_epoch 300 \
                --target_transform
        done
    done
done

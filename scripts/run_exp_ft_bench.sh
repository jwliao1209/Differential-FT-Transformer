#!/bin/bash

data_ids=(
    # adult
    aloi
    california_housing
    covtype
    epsilon
    helena
    higgs_small
    jannis
    microsoft
    yahoo
    year
)

models=(
    dofen
    ft
    diff
    dint
)

for data_id in ${data_ids[@]}; do
    for model in ${models[@]}; do
        python run.py \
            --data_dir ft_transformer_benchmark \
            --data_id $data_id \
            --model $model \
            --n_epoch 300 \
            --target_transform
    done
done

# Differential-FT-Transformer

This repository implement the Differential Transformer for tabluar data.


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 dftt
source dftt/bin/activate
pip install -r requirements.txt
```

## FT-Transformer Benchmark
Run the following command to download dataset:
```
wget https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1 -O revisiting_models_data.tar.gz
tar -xvf revisiting_models_data.tar.gz
```


## Training
To train the model, run the command:
```
bash scripts/run_exp.sh
```


## Unit Test
To excute the unit test, run the command:
```
bash scripts/run_test.sh
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{liao2024_dftt,
    title  = {Differential FT-Transformer},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Differential-FT-Transformer},
    year   = {2025}
}

@article{cang2025dint,
    title   = {DINT Transformer},
    author  = {Yueyang Cang, Yuhang Liu, Xiaoteng Zhang, Erlu Zhao, Li Shi},
    journal = {arXiv preprint arXiv:2501.17486},
    year    = {2025}
}

@article{zhu2025transformers,
    title   = {Transformers without Normalization},
    author  = {Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu},
    journal = {arXiv preprint arXiv:2503.10622},
    year    = {2025}
}

@inproceedings{ye2025differential,
    title     = {Differential transformer},
    author    = {Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2025}
}

@inproceedings{gorishniy2021revisiting,
    title     = {Revisiting Deep Learning Models for Tabular Data},
    author    = {Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko},
    booktitle = {Annual Conference on Neural Information Processing Systems (NeurIPS)},
    year      = {2021},
}
```

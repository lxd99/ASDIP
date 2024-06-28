# Adaptive Sampling-based Dynamic Graph Learning for Information Diffusion Prediction

This is the implementation of ASDIP: [Adaptive Sampling-based Dynamic Graph Learning for Information Diffusion Prediction].

## Requirements

- python == 3.9.18
- pytorch == 2.0.1
- numba == 0.55.1
- numpy == 1.21.6
- pandas == 2.0.3
- scikit-learn == 1.3.0
- dgl == 1.1.2
- tqdm 
- wandb

## Dataset

- Download the preprocessed dataset from [Baidu Yun](https://pan.baidu.com/s/1H_3JtoOKEiW1c0GPsgb6Nw ) （extract code nihb）
- create a directory `./data/dynamic` and put the downloaded dataset into the directory.

## Run
Create the directories to store the running results 
```sh
mkdir log results saved_models
```
Running command

```sh
#Memetracker
python train_ASDIP.py --prefix std --dataset douban --gpu 0 --causal before after none
#Douban
python train_ASDIP.py --prefix std --dataset memetracker --gpu 0 --causal before after none
#Weibo
python train_ASDIP.py --prefix std --dataset weibo --gpu 0 --causal before after none
```




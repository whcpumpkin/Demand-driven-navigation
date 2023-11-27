# Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/demand-driven-navigation)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2309.08138)

This repo is the official implementation of NeurIPS 2023 paper, [Demand-driven Navigation](https://arxiv.org/abs/2309.08138)

## TODOs (Under Development):
- [x] README
- [x] Instruction Dataset
- [x] Trajectory Dataset
- [ ] Pre-generated Dataset
- [ ] Utils Code
- [ ] Testing
- [ ] Training

## Warning: I am currently refactoring my code. Although all code committed to the repository has been tested, there may still be some minor issues.


## Overview
<img src="demo/NIPS-2023-DDN.gif" align="middle" width="700"/> 
We propose a demand-driven navigation task, which requires an agent to find objects that satisfy human demands, and propose a novel method to solve this task.

## Materials Download (Under Updating)

For dataset and pretrained models, the download link is [here](https://drive.google.com/drive/folders/1iR-zf3SHLMhA05IQXsQGUfyfB-8spFC-?usp=sharing).

For Chinese, we provide [百度网盘](https://pan.baidu.com/s/1ghLdUjp5AMCTqpLOM1byVw?pwd=1rid).

## Dataset

### Instruction Dataset
Please see [dataset](./dataset/).

### Trajectory Dataset

We provide the raw trajectory data. Please move them to [dataset](./dataset/) and then unzip them. The following is the structure of the files in the `raw_trajectory_dataset.zip` package. `bc_{train,val}_check.json` are the metadata of trajectory dataset.

```
┌bc
│ ├train
│ │  └house_{idx}
│ │      └path_{idx}
│ │         └{idx}.jpg
│ └val
│    └house_{idx}
│        └path_{idx}
│ │         └{idx}.jpg
├bc_train_check.json
┕bc_val_check.json
```

### Pre-generated Dataset

In order to speed up the training, we use DETR model to segment the image in advance and get the corresponding CLIP-Visual-Feature. It takes $30h$ in a server with dual E5-2680V4 processors and a 22GB RTX 2080Ti graphics card.

```
python generate_pre_data.py --mode=pre_traj_crop --dataset_mode=train --top_k=16 
python generate_pre_data.py --mode=pre_traj_crop --dataset_mode=val --top_k=16 
python generate_pre_data.py --mode=merge_pre_crop_json 

```

### Training

To train the Attribute Module, Prepare the following files in the [dataset](./dataset/): `instruction_{train,val}_check.json`, `LGO_features.json`, `instruction_bert_features.json`

Then run:

```
python train_attribute_features.py --epoch=5000
```


## Contact
If you have any suggestion or questions, please feel free to contact us:

[Hongcheng Wang](https://whcpumpkin.github.io): [whc.1999@pku.edu.cn](mailto:whc.1999@pku.edu.cn)

[Hao Dong](https://zsdonghao.github.io/): [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn)
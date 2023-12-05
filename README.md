# Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/demand-driven-navigation)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2309.08138)

This repo is the official implementation of NeurIPS 2023 paper, [Demand-driven Navigation](https://arxiv.org/abs/2309.08138)

## TODOs (Under Development):
- [x] README
- [x] Instruction Dataset
- [x] Trajectory Dataset
- [x] Pre-generated Dataset
- [x] Utils Code
- [x] Training
- [x] Testing

## Warning: I am currently refactoring my code. Although all code committed to the repository has been tested, there may still be some minor issues. More comments will be continuously added to the code to improve its readability. Feel free to ask any questions.


## Overview
<img src="demo/NIPS-2023-DDN.gif" align="middle" width="700"/> 
We propose a demand-driven navigation task, which requires an agent to find objects that satisfy human demands, and propose a novel method to solve this task.

## Materials Download (Under Updating)

For all dataset and pretrained models, the download link is [Googledrive](https://drive.google.com/drive/folders/1iR-zf3SHLMhA05IQXsQGUfyfB-8spFC-?usp=sharing) and [Onedrive](https://chinapku-my.sharepoint.com/:f:/g/personal/1800012939_pku_edu_cn/EpUlnqhbNflHvDbA-fG6h94BEsfP9KE6FaWDFKe3g3xXMQ?e=g1DabS)(recommend).

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

We have provided the pre-generated dataset in the `Materials Download`.


### Training

#### Attribute Module

To train the Attribute Module, prepare the following files in the [dataset](./dataset/): `instruction_{train,val}_check.json`, `LGO_features.json`, `instruction_bert_features_check.json`

Then run:

```
python train_attribute_features.py --epoch=5000
```
Finally, select the model with the lowest loss on the validation set, named `attribute_model2.pt`.

#### Navigation Policy

To train the navigation policy, prepare the following files in the [dataset](./dataset/): `bc_train_{0,1,2,3,4}_pre.h5`, `bc_{train,val}_check.json`, in the [pretrained_model](./pretrained_model/): `attribute_model2.pt`, `mae_pretrain_model.pth`

Then run
```
python main.py --epoch=30 --mode=train_DDN  --workers=32 --dataset_mode=train --device=cuda:0
```


### Testing

#### Model Selection

First, we need to select the model using validation set. 
```
python eval.py --mode=eval_DDN --eval_path=$path_to_saved_model$ --dataset_mode=val  --device=cuda:0 --workers=32
```
Then we select the model with the highest accuracy on the validation set, assuming its index is $idx$.

#### Navigation Policy Testing

```
python eval.py --mode=test_DDN --eval_path=$path_to_saved_model$ --dataset_mode=$train,test$ --seen_instruction=$0,1$  --device=cuda:0 --epoch=500 --eval_ckpt=$idx$
```

For the parameter `dataset_mode`, 'train' represents 'seen_scene', while 'test' represents 'unseen_scene'. Just choose one of them during the test.

For the parameter `seen_instruction`, '1' represents 'seen_instruction', while '0' represents 'unseen_scene'. Just choose one of them during the test.

Note: if you run AI2Thor in a headless machine, `xvfb` is highly recommended. Here is an example.
```
xvfb-run -a python eval.py --mode=test_DDN --eval_path=$path_to_saved_model$ --dataset_mode=train --seen_instruction=1  --device=cuda:0 --epoch=500 --eval_ckpt=15
```

## Contact
If you have any suggestion or questions, please feel free to contact us:

[Hongcheng Wang](https://whcpumpkin.github.io): [whc.1999@pku.edu.cn](mailto:whc.1999@pku.edu.cn)

[Hao Dong](https://zsdonghao.github.io/): [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn)
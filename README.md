# Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/demand-driven-navigation)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2309.08138)

This repo is the official implementation of NeurIPS 2023 paper, [Demand-driven Navigation](https://arxiv.org/abs/2309.08138)

## TODOs (Under Development):
- [x] README
- [x] Instruction Dataset
- [ ] Trajectory Dataset
- [ ] Utils Code
- [ ] Testing
- [ ] Training
- [ ] Refine and Vis

## Overview
<img src="demo/NIPS-2023-DDN.gif" align="middle" width="700"/> 
We propose a demand-driven navigation task, which requires an agent to find objects that satisfy human demands, and propose a novel method to solve this task.

## Dataset

### Instruction Dataset
Please see [dataset](./dataset/).

### Trajectory Dataset
The download link is [here](https://drive.google.com/file/d/1xcI5j6AHx3MCNjzhpWrtM_7B06KPL_6_/view?usp=sharing).

For Chinese, we provide [百度网盘](https://pan.baidu.com/s/1ghLdUjp5AMCTqpLOM1byVw?pwd=1rid).

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



## Contact
If you have any suggestion or questions, please feel free to contact us:

[Hongcheng Wang](https://whcpumpkin.github.io):[whc.1999@pku.edu.cn](mailto:whc.1999@pku.edu.cn)

[Hao Dong](https://zsdonghao.github.io/): [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn)
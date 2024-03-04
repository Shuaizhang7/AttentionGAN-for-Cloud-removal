# Cloud removal using SAR and optical images via attention mechanism-based GAN
This repository provides the PyTorch code for our paper “Cloud removal using SAR and optical images via attention mechanism-based GAN”. This code is based on the implementation of [Attention-GAN](https://github.com/xinyuanc91/Attention-GAN) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The pipline of our method:
![图片](/fig/pipline.png "method pipline")
The result of our method:
![图片](/fig/result.png "result")
Cloud removal results in the real data experiment. Each row corresponds to an example of the results. Column (a) shows the SAR images. Column (b) shows the attention map. Column (c) shows the cloud images. Columns (d) to (h) correspond to the results obtained by the pix2pix model, the SAR-opt-GAN model, the Simulation-Fusion GAN model, the GLF-CR model, and the proposed model. Column (i) shows the ground truth images.

## Data
我们使用[SEN12MS-CR数据集]https://patricktum.github.io/cloud_removal/sen12mscr/
The directory structure of the data is as follows:
```
data
│
└───train/test
│   │
│   └───trainA/testA  #cloud images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│   │
│   └───trainB/testB  #cloudless images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│   │
│   └───trainC/testC  #SAR images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│ 
```
## Environment
You can Install dependencies via:
```bash
conda env create --file environment.yml
```
or:
```bash
pip install -r requirements.txt
```

## Train
    sh scripts/train.sh
## Test
    sh scripts/test.sh

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper.
```
@article{zhang2023cloud,
  title={Cloud removal using SAR and optical images via attention mechanism-based GAN},
  author={Zhang, Shuai and Li, Xiaodi and Zhou, Xingyu and Wang, Yuning and Hu, Yue},
  journal={Pattern Recognition Letters},
  volume={175},
  pages={8--15},
  year={2023},
  publisher={Elsevier}
}
```

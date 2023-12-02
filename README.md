# Correspondence attention for facial appearance simulation

## Introduction
In this work, we have formulated an ACMT-Net incorporating a novel CPSA module. ACMT-Net is designed to accurately predict the change of one point set prompted by the movement of another point set. We further proposed a novel k-NN-based contrastive learning approach for pre-training the attentive correspondence between bony and facial point sets, enhancing its capability to model spatial correspondence. The proposed ACMT-Net attains the same level of accuracy as the state-of-the-art FEM simulation method, while considerably reducing the computational time required during the surgical planning processes. For more details on the network, please refer to our [MICCAI 2022 paper](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_54) or the pre-print version available on [arXiv](https://arxiv.org/pdf/2210.01685.pdf).

## Demo
![Demo](demo/demo.gif)

**Prerequisites**
- Linux (tested under Ubuntu 16.04 )
- Python (tested under 2.7)
- TensorFlow (tested under 1.4.0-GPU )
- numpy, h5py

The code is built on the top of [PointNET++](https://github.com/charlesq34/pointnet2). 
Before run the code, please compile the customized TensorFlow operators of PointNet++ under the folder "/Prediction_net/tf_ops".

**Train and test**

To trian a model:

`python -u run.py --mode=train  --gpu=0`

To test the trained model:

`python -u run.py --mode=test  --gpu=0`

Note that the train and test hdf5 files have been set in the program. If you have errors when running this code please check ALL the path first.

## Citation
**Conference version
```
@inproceedings{fang2022deep,
  title={Deep Learning-Based Facial Appearance Simulation Driven by Surgically Planned Craniomaxillofacial Bony Movement},
  author={Fang, Xi and Kim, Daeseung and Xu, Xuanang and Kuang, Tianshu and Deng, Hannah H and Barber, Joshua C and Lampen, Nathan and Gateno, Jaime and Liebschner, Michael AK and Xia, James J and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={565--574},
  year={2022},
  organization={Springer}
}
```


[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)

# Keyword Spotting ConvMixer - Official PyTorch Implementation

## Introduction - ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-field Keyword Spotting

This repository demonstrates the building of a small-footprint keyword spotting model. The architecture, convMixer for keyword spotting, is based on the accepted ICASSP 2022 [paper](https://arxiv.org/abs/2201.05863). Our code uses Pytorch and the requirements are listed in this project.

## Abstract
Building efficient architecture in neural speech processing is paramount to success in keyword spotting deployment. However, it is very challenging for lightweight models to achieve noise robustness with concise neural operations. In a real-world application, the user environment is typically noisy and may contain reverberations. We proposed a novel feature interactive convolutional model with merely 100K parameters to tackle this under the noisy far-field condition. The interactive unit is proposed in place of the attention module that promotes the flow of information with more efficient computations. Moreover, curriculum-based multi-condition training
is adopted to attain better noise robustness. Our model achieves 98.2% top-1 accuracy on Google Speech Command V2-12 and is competitive against large transformer models under the designed noise condition.

## Requirements

1. Python version 3.7, 3.8
2. Pytorch 1.8.1 and above

## Installation

You can install the requirements using pip.

`pip install -r requirements.txt`

## Code pipeline
To build on Google Speech Command dataset, you can exploit the script in `data_prep/process_speech_commands_data.py` to download the dataset. Once completed, you may want to move the manifest files to the folder `manifests` for convenience. Otherwise build custom data model with the required `manifest.json` following the format dictionary for each sample

`{"audio_filepath": audio_path, "duration": duration, "command": label}`

To train, check the validity of the path for manifest_filepath in `configs/ConvMixer.yaml`. Following that, run the command below to start training.

python main.py

NOTE: This current version is built for original clean Google Speech Command. We will release the setup for far-field noisy keyword spotting in the future. 

## TODO
Update this repository with curriculum learning for noisy far-field keyword spotting.

## Citation

```
@article{ng2022convmixer,
  title={ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-field Keyword Spotting},
  author={Ng, Dianwen and Chen, Yunqi and Tian, Biao and Fu, Qiang and Chng, Eng Siong},
  journal={arXiv preprint arXiv:2201.05863},
  year={2022}
}
```
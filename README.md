
## Installation

#### ONNX
[[pytorch18]](https://drive.google.com/file/d/1vVSep78Xdf_LuqxwsCvuX5Ue_ajAkckf/view?usp=sharing),
[[pytorch110]](https://drive.google.com/file/d/13aIQvXkdcZ_vZMm2x-kQ0CU4mRtE2H2q/view?usp=sharing)

#### 1. Create a conda environment with python 3.8 (3.9 for pytorch110):

```shell
conda create -n <env_name> python=3.8
conda activate <env_name>
```

~~#### 2. We use pytorch 1.9.0, torchvision 0.10.0, and cudatoolkit 11.1:~~
#### 2. We use pytorch:
Please refer to https://pytorch.org/get-started/locally/ for installing.

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
or
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia
```


#### 3. Use pip to install some requirements: 

```shell
pip install matplotlib easydict tqdm lmdb tensorboard
pip install transformers  # for BERT https://huggingface.co/transformers/
pip install opencv_python
```

#### 4. We use [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation: 
```shell
pip install -U albumentations
```

#### 5. Install TensorRT-8.0.1.6

[Tar File Installation
](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)


---

## ~~Evaluation~~

---

## ~~Training~~


---

## Acknowledgments
Thanks for the great [DETR](https://github.com/facebookresearch/detr) and [pytracking](https://github.com/visionml/pytracking).
- We borrow `transformer` from [DETR](https://github.com/facebookresearch/detr).
- We refer to [pytracking](https://github.com/visionml/pytracking) for data augmentation.

## License
Our work is released under the GPL 3.0 license. Please see the LICENSE file for more information.


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

## Evaluation

#### 1. Download our pre-trained model weights:
The pre-trained models are available in [Gdrive](https://drive.google.com/file/d/19qVDFydekUHW75YzF4AkA2xQcg3Vh3v2/view?usp=sharing). Download the file and place it in `<./checkpoints>`.


#### 2. We provide a multi-process testing script for evaluation on several benchmarks:
```shell
# --benchmark ('tnl2k', 'lasot', 'otb99lang', 'vot2019lt')
python test.py --num_gpu=2 --num_process=6 --experiment=multimodal --train_set=language --benchmark=tnl2k --vis
``` 
For VOT long-term dataset, we provide the langauge annotations (category name) in `<./LTB50_language>`, which is used for our experiment.

#### 3. Ablation study:

Please check `Line118~138` in `<lib/model/model_multimodal.py>` and `Line104~111` in `<lib/model/heads/extreme_V3.py>`.
Comment out the corresponding lines and run the testing script.

#### 4. Raw results:

The raw results can be downloaded from [Gdrive](https://drive.google.com/file/d/1tW5-VgDkiNTl3xvRQAyNoWhsr-lhlrHD/view?usp=sharing).

---

## Training

#### 1. Prepare the training data:

***(We use LMDB to store the training data. We will upload our LMDB file or release the related code if our paper is accepted.)***

#### 2. Train the model:
We recommend training in `DistributedDataParallel` mode. 
All of our models are trained on a single machine with two RTX3090 GPUs.
It takes about two days to train our model.

For distributed training on a single node with 2 GPUs:
```shell
python -m torch.distributed.launch --nproc_per_node=2  train.py --gpu_id=0,1 --experiment=multimodal --train_set=language
```

For distributed training on 2 nodes with 2 GPUs per node:
```shell
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=10.7.9.26 --master_port=1234 train.py --gpu_id=0,1 --experiment=multimodal --train_set=language
```

You can also run the training script directly to train or debug with single GPU:
```shell
python train.py --gpu_id=0,1 --experiment=multimodal --train_set=language
```



---

## Acknowledgments
Thanks for the great [DETR](https://github.com/facebookresearch/detr) and [pytracking](https://github.com/visionml/pytracking).
- We borrow `transformer` from [DETR](https://github.com/facebookresearch/detr).
- We refer to [pytracking](https://github.com/visionml/pytracking) for data augmentation.

## License
Our work is released under the GPL 3.0 license. Please see the LICENSE file for more information.

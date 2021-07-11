# VSR-guided-CIC

Official codebase for paper _[Human-like Controllable Image Captioning with Verb-specific Semantic Roles](https://arxiv.org/abs/2103.12204)_ (CVPR 2021).

## 1. Prerequisites

The following dependencies should be enough. See [vsr.yml](vsr.yml) for more environment settings.

```
h5py 2.10.0
python 3.6.10
pytorch 1.5.1
munkres 1.0.12
numpy 1.18.5
speaksee 0.0.1
tensorboardx 2.0
torchvision 0.6.1
tqdm 4.46.0
```

## 2. Data Preparation

### *2.1 Semantic Role Label*

Install semantic role labeling tool with `pip install allennlp==1.0.0 allennlp-models==1.0.0`.
The model we used is provided as [bert-base-srl-2020.03.24.tar.gz](https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz). The latest version of this tool can be found in [here](https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz).
You can follow the demo in [AllenNLP_demo](https://demo.allennlp.org/semantic-role-labeling). And we will release the [demo](demo/srl_use.ipynb) to show how to process data with this semantic role labeling tool.

### *2.2 Pretrained Models & Processed Data*

The models can be downloaded from [here](https://drive.google.com/file/d/1YyCMntSqeGxH3QKqJlqG5S_YLd3UmoSc/view?usp=sharing) and extracted into "./saved_model".
And other data can be downloaded from [link1](https://drive.google.com/file/d/14-QZqLqv7QafAzOfatLxaOx1B6zmgrZO/view?usp=sharing), [link2](https://drive.google.com/file/d/1bi3kBnb_xHzX2EaDwdsQHXS48u_Qd3ow/view?usp=sharing) and extracted into "./datasets" and "./saved_data", repectively.
(using `tar -xzvf *.tgz`)

The process code to generate those data will be released soon.

*Flickr30k Entities*: The detection feature can be downloaded from [here](https://drive.google.com/file/d/1uQ7n_Pr51Kretej05ZNGYdM8g26XhwKa/view?usp=sharing) and extracted into './datasets/flickr/flickr30k_detections.hdf5'.

*COCO Entities*: The detection feature can be downloaded from [here](https://drive.google.com/file/d/1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx/view?usp=sharing) and extracted into './datasets/coco/coco_detections.hdf5'. Refer to the [show, control and tell](https://github.com/aimagelab/show-control-and-tell) for more information about *COCO Entities*.

## 3. Training

### *3.1 Train GSRL Model*

We will release training code later.

Train GSRL model on two datasets: Flickr30k Entities and COCO Entities.

### *3.2 Train S-level & R-level SSP*

***S-level SSP***

```bash
# train s-level SSP on Flickr30k Entities
python flickr_scripts/train_region_sort_flickr.py --checkpoint_path saved_model/flickr_s_ssp

# train s-level SSP on COCO Entities
python coco_scripts/train_region_sort.py --checkpoint_path saved_model/coco_s_ssp
```

***R-level SSP***

```bash
# train sinkhorn model on Flickr30k Entities
python flickr_scripts/train_sinkhorn_flickr.py --checkpoint_path saved_model/flickr_sinkhorn

# train sinkhorn model on COCO Entities
python coco_scripts/train_sinkhorn.py --checkpoint_path saved_model/coco_sinkhorn
```

### *3.3 Train Role-shift Captioning Model*

Firstly, train the captioning model with XE(cross-entropy), with 

```bash
python coco_scripts/train.py --exp_name captioning_model --batch_size 100 --lr 5e-4
```

Next, further train it with RL with CIDEr reward, with 

```bash
python coco_scripts/train.py --exp_name captioning_model --batch_size 100 --lr 5e-5 --sample_rl
```

## 4. Evaluation

| Argument | Values |
|------|------|
| --det | whether use detection regions or use gt regions |
| --gt | whether use gt verb or predicted verb |

### *Evaluate on Flickr30k Datasets*

```bash
python flickr_scripts/eval_flickr.py
python flickr_scripts/eval_flickr.py --gt
python flickr_scripts/eval_flickr.py --det
python flickr_scripts/eval_flickr.py --gt --det
```

### *Evaluate on MSCOCO Datasets*

```bash
python coco_scripts/eval_coco.py
python coco_scripts/eval_coco.py --gt
python coco_scripts/eval_coco.py --det
python coco_scripts/eval_coco.py --gt --det
```

## Contact & Cite

If there is any question about our work, please contact us with [issue](https://github.com/mad-red/VSR-guided-CIC/issues) or email us with zju_jiangzhihong@zju.edu.cn.

Please cite with the following Bibtex:

```
@inproceedings{chen2021vsr,
  title={Human-like Controllable Image Captioning with Verb-specific Semantic Roles},
  author={Chen, Long and Jiang, Zhihong and Xiao, Jun and Liu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

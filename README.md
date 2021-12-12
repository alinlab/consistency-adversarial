# Consistency Regularization for Adversarial Robustness

Official PyTorch implementation of [Consistency Regularization for Adversarial Robustness](https://arxiv.org/abs/2103.04623) (AAAI 2022) by 
[Jihoon Tack](https://jihoontack.github.io), 
[Sihyun Yu](https://sihyun-yu.github.io), 
[Jongheon Jeong](https://sites.google.com/view/jongheonj), 
[Minseon Kim](https://kim-minseon.github.io), 
[Sung Ju Hwang](http://www.sungjuhwang.com/),
and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).


## 1. Dependencies
```
conda create -n con-adv python=3
conda activate con-adv

conda install pytorch torchvision cudatoolkit=11.0 -c pytorch 

pip install git+https://github.com/fra31/auto-attack
pip install advertorch tensorboardX
```

## 2. Training
### 2.1. Training option and description
The option for the training method is as follows:
- `<DATASET>`: {`cifar10`,`cifar100`,`tinyimagenet`}
- `<AUGMENT>`: {`base`,`autoaug`,`autoaug_sche`}
- `<ADV_TRAIN OPTION>`: {`adv_train`,`adv_trades`,`adv_mart`}

Current code are assuming l_infinity constraint adversarial training and PreAct-ResNet-18 as a base model.\
To change the option, simply modify the following configurations:
- WideResNet-34-10: `--model wrn3410`
- l_2 constraint: `--distance L2`


### 2.2. Training code
#### Standard cross-entropy training
```
# Standard cross-entropy
python train.py --mode ce --augment_type base --dataset <DATASET>
```

#### Adversarial training
```
# Adversarial training
python train.py --mode <ADV_TRAIN OPTION> --augment_type <AUGMENT> --dataset <DATASET>

# Example: Standard AT under CIFAR-10
python train.py --mode adv_train --augment_type base --dataset cifar10
```

#### Consistency regularization
```
# Consistency regularization
python train.py --consistency --mode <ADV_TRAIN OPTION> --augment_type <AUGMENT> --dataset <DATASET>

# Example: Consistency regularization based on standard AT under CIFAR-10
python train.py --consistency --mode adv_train --augment_type autoaug_sche --dataset cifar10 
```

## 3. Evaluation
### 3.1. Evaluation option and description
The description for treat model is as follows:
- `<DISTANCE>`: {`Linf`,`L2`,`L1`}, the norm constraint type
- `<EPSILON>`: the epsilon ball size
- `<ALPHA>`: the step size of PGD optimization
- `<NUM_ITER>`: iteration number of PGD optimization

### 3.2. Evaluation code
#### Evaluate clean accuracy
```
python eval.py --mode test_clean_acc --dataset <DATASET> --load_path <MODEL_PATH>
```

#### Evaluate clean & robust accuracy against PGD
```
python eval.py --mode test_adv_acc --distance <DISTANCE> --epsilon <EPSILON> --alpha <ALPHA> --n_iters <NUM_ITER> --dataset <DATASET> --load_path <MODEL_PATH>
```

#### Evaluate clean & robust accuracy against AutoAttack
```
python eval.py --mode test_auto_attack --epsilon <EPSILON> --distance <DISTANCE> --dataset <DATASET> --load_path <MODEL_PATH>
```

#### Evaluate mean corruption error (mCE)
```
python eval.py --mode test_mce --dataset <DATASET> --load_path <MODEL_PATH>
```

## Citation
```
@inproceedings{tack2022consistency,
  title={Consistency Regularization for Adversarial Robustness},
  author={Jihoon Tack and Sihyun Yu and Jongheon Jeong and Minseon Kim and Sung Ju Hwang and Jinwoo Shin},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

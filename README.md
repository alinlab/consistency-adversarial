# Consistency Regularization for Adversarial Robustness

Official PyTorch implementation of Consistency Regularization for Adversarial Robustness by 
[Jihoon Tack](https://jihoontack.github.io), 
[Sihyun Yu](https://sihyun-yu.github.io), 
[Jongheon Jeong](), 
[Minseon Kim](https://kim-minseon.github.io), 
[Sung Ju Hwang](http://www.sungjuhwang.com/),
and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html).

<p align="center">
    <img src=figures/augmentation.png width="900"> 
</p>


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
- `<AUGMENT>`: {`base`,`ccg`}
- `<ADV_TRAIN OPTION>`: {`adv_train`,`adv_trades`,`adv_mart`}

Current code are assuming l_infinity constraint adversarial training and PreAct-ResNet-18 as a base model.\
To change the option, simply modify the following configurations:
- WideResNet-34-10: `--model wrn3410`
- l_2 constraint: `--distance L2`


### 2.2. Training code
#### Standard cross-entropy training
```
% Standard cross-entropy
python train.py --mode ce --augment base --dataset <DATASET>
```

#### Adversarial training
```
% Adversarial training
python train.py --mode <ADV_TRAIN OPTION> --augment <AUGMENT> --dataset <DATASET>

% Example: Standard AT under CIFAR-10
python train.py --mode adv_train --augment base --dataset cifar10
```

#### Consistency regularization
```
% Consistency regularization
python train.py --consistency --mode <ADV_TRAIN OPTION> --augment <AUGMENT> --dataset <DATASET>

% Example: Consistency regularization based on standard AT under CIFAR-10
python train.py --consistency --mode adv_train --augment ccg --dataset cifar10 
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

## 4. Results
### White box attack

Clean accuracy and robust accuracy (%) against white-box attacks on PreAct-ResNet-18 trained on CIFAR-10.\
We use l_infinity threat model with epsilon = 8/255.

| Method               | Clean  | PGD-20 | PGD-100 | AutoAttack |
| ---------------------|--------|--------|---------|------------|
| Standard AT          | 84.48  |  46.09 |  45.89  |    40.74   | 
| + Consistency (Ours) | 84.65  |  54.86 |  54.67  |    47.83   |
| TRADES               | 81.35  |  51.41 |  51.13  |    46.41   |
| + Consistency (Ours) | 81.10  |  54.86 |  54.68  |    48.30   |
| MART                 | 81.35  |  49.60 |  49.41  |    41.89   |
| + Consistency (Ours) | 81.10  |  55.31 |  55.16  |    47.02   |

### Unseen adversaries
Robust accuracy (%) of PreAct-ResNet-18 trained with of l_infinity epsilon = 8/255 constraint against unseen attacks.\
For unseen attacks, we use PGD-100 under different sized l_infinity epsilon balls, and other types of norm balls.

| Method               | l_infinity, eps=16/255  | l_2, eps=300/255 | l_1, eps=4000/255 |
| ---------------------|--------|--------|---------|
| Standard AT          | 15.77  |  26.91 |  32.44  |
| + Consistency (Ours) | 22.49  |  34.43 |  42.45  |
| TRADES               | 23.87  |  28.31 |  28.64  |
| + Consistency (Ours) | 27.18  |  37.11 |  46.73  |
| MART                 | 20.08  |  30.15 |  27.00  |
| + Consistency (Ours) | 27.91  |  38.10 |  43.29  |

### Mean corruption error
Mean corruption error (mCE) (%) of PreAct-ResNet-18 trained on CIFAR-10, and tested with CIFAR-10-C dataset

| Method               | mCE    | 
| ---------------------|--------|
| Standard AT          | 24.05  |
| + Consistency (Ours) | 22.06  |
| TRADES               | 26.17  |
| + Consistency (Ours) | 24.05  |
| MART                 | 27.75  |
| + Consistency (Ours) | 26.75  |

## Reference
- [TRADES](https://github.com/yaodongyu/TRADES)
- [MART](https://github.com/YisenWang/MART)

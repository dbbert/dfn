# Introduction

This repository contains code to reproduce the experiments in [Dynamic Filter Networks](https://arxiv.org/pdf/1605.09673v2.pdf), a NIPS 2016 paper by Bert De Brabandere\*, Xu Jia\*, Tinne Tuytelaars and Luc Van Gool (\* Bert and Xu contributed equally).

In a traditional convolutional layer, the learned filters stay fixed after training. In contrast, we introduce a new framework, the Dynamic Filter Network, where filters are generated dynamically conditioned on an input.

Example:

![mnist prediction](https://i.imgur.com/XbyD2ix.png)

<!---
![mnist gif1](https://i.imgur.com/vmkSn0k.gif)
![mnist gif2](https://i.imgur.com/JzGhE31.gif)
-->

If you use our code in your research, please cite following paper:
```
@inproceedings{debrabandere16dynamic,
  author = {De Brabandere, Bert and Jia, Xu and Tuytelaars, Tinne and Van Gool, Luc},
  title = {Dynamic Filter Networks},
  booktitle = {NIPS},
  year = {2016}
}
```

# Running the code

* Install [Lasagne](https://lasagne.readthedocs.io/en/latest/user/installation.html) and its prerequisites.
* [Download the datasets](https://drive.google.com/file/d/1p7K9FBjQBwAbH4UOdLYy2FFBBDsTpF2g/view?usp=sharing) and update the paths in the datasets/dataset_*.py files to point to them.
* Run the experiments:
```
python experiment_steerableFilter.py
python experiment_bouncingMnistOriginal.py
python experiment_highwayDriving.py
python experiment_stereoPrediction.py
```
This will write checkpoint files to the checkpoints directory.

* You can also run the baseline models. They have the same architecture as the DFN models, but without the DFN layer at the end:
```
python experiment_bouncingMnistOriginal_baseline.py
python experiment_highwayDriving_baseline.py
python experiment_stereoPrediction_baseline.py
```
Finally, you can evaluate the DFN and baseline models on the test set and generate new predictions with the notebook files:
```
analyse_trained.ipynb
analyse_trained-baseline.ipynb
```

## Tensorflow implementation
A tensorflow implementation of the steerable filter experiment is available in `experiment_steerableFilter_tensorflow`. We have also added a basic tensorflow implementation of the bouncing mnist video prediction experiment in [experiment_bouncingMnistOriginal_tensorflow.ipynb](https://github.com/dbbert/dfn/blob/master/experiment_bouncingMnistOriginal_tensorflow.ipynb). Only training is implemented, no evaluation on the test set. There might be some small differences compared to the Lasagne implementation, so we cannot guarantee that you reach the same accuracy that is reported in the paper.  

# Results

When evaluating the trained models on the test sets with the ipython notebooks, you should approximately get following results:

| Loss (per pixel)      | Baseline | DFN       |
| --------------------- |:--------:| ---------:|
| Moving MNIST (bce)    | 0.106144 | 0.068914  |
| Highway Driving (mse) | 0.003683 | 0.003270  |
| Stereo Cars (mse)     | 0.000416 | 0.000330  |

| Loss (image, 64x64)   | Baseline | DFN      |
| --------------------- |:--------:| --------:|
| Moving MNIST (bce)    | 434.8    | 282.3    |
| Highway Driving (mse) | 15.08    | 13.39    |
| Stereo Cars (mse)     | 1.70     | 1.35     |

| # Params        | Baseline  | DFN     |
| --------------- |:---------:|:-------:|
| Moving MNIST    | 637,443   | 637,361 |
| Highway Driving | 368,245   | 368,122 |
| Stereo Cars     | 464,509   | 464,494 |

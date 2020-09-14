# Using Soft Actor-Critic for Low-Level UAV Control

This repository is the official implementation of [Using Soft Actor-Critic for Low-Level UAV Control](). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
./training.sh
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model with the optimal policy, run:

```eval
./evaluate.sh
```

<!-- >📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

## Pre-trained Models

You can check the saved trained policies in:

- [saved_policies/](saved_policies/) 

<!-- >📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## Credits

Code heavily based in [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

Environment is a continuation of the work in:

@article{CANOLARS,
author = {Lopes, Guilherme and Ferreira, Murillo and Simões, Alexandre and Colombini, Esther},
title = {Intelligent Control of a Quadrotor with Proximal Policy Optimization Reinforcement Learning},
year = {2018},
month = {11},
journal = {Latin American Robotic Symposium},
pages = {503-508},
doi = {10.1109/LARS/SBR/WRE.2018.00094}
}

## Results

Run the notebooks on [notebooks/](notebooks/) to check the trajectories presented on the paper. 

<!-- ### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
 -->

## License

[MIT-LICENSE](License.md)

## Cite us


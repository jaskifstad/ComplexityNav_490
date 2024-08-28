# Characterizing the Complexity of Social Robot Navigation Scenarios
This repository contains the code for our [paper](https://arxiv.org/abs/2405.11410), Characterizing the Complexity of Social Robot Navigation Scenarios. The repo is built off of [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning) (which itself is built off of the original [CrowdNav](https://github.com/vita-epfl/CrowdNav)), however we added features including introduction of static obstacles, handling of multiple pedestrian policies in a single scenario, and the directionalities outlined in the paper. We have also integrated large sections of code from the [Pred2Nav](https://github.com/sriyash421/Pred2Nav) repository and made use of the [PytorchMPPI](https://github.com/UM-ARM-Lab/pytorch_mppi) repository to add support for model predictive control methods.


## Abstract
Social robot navigation algorithms are often demonstrated in overly simplified scenarios, prohibiting the extraction of practical insights about their relevance to real-world domains. Our key insight is that an understanding of the inherent complexity of a social robot navigation scenario could help characterize the limitations of existing navigation algorithms and provide actionable directions for improvement. Through an exploration of recent literature, we identify a series of factors contributing to the Complexity of a scenario, disambiguating between contextual and robot-related ones. We then conduct a simulation study investigating how manipulations of contextual factors impact the performance of a variety of navigation algorithms. We find that dense and narrow environments correlate most strongly with performance drops, while the heterogeneity of agent policies and directionality of agents have a less pronounced effect. This motivates a shift towards developing and testing algorithms under higher Complexity settings.

## Setup
We used Python 3.8.10 and Pytorch 1.13 for development and experiments.
First follow the basic setup for [RelationalGraphLearning](https://github.com/ChanganVR/RelationalGraphLearning):

1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

Then ensure the dependencies for [Pred2Nav](https://github.com/sriyash421/Pred2Nav) and [PytorchMPPI](https://github.com/UM-ARM-Lab/pytorch_mppi) are satisfied.

## Getting Started
The repository is organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework and method training and tuning details from the paper can be found
[here](crowd_sim/README.md).


1. Train a policy.
```
python train.py --policy rgl
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy rgl --model_dir data/output --phase test --visualize
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```

To run a sequence of experiments, first define a config file with the desired parameters. Then run the following:
```
python test.py --policy rgl --model_dir data/output --phase test
```
The config file in the icra_benchmark directory is set up by default to reproduce the experiments from the paper.


## Citation
If you find the code or paper useful for your research, please cite the following paper:
```bibtex
@misc{stratton2024characterizingcomplexitysocialrobot,
      title={Characterizing the Complexity of Social Robot Navigation Scenarios}, 
      author={Andrew Stratton and Kris Hauser and Christoforos Mavrogiannis},
      year={2024},
      eprint={2405.11410},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2405.11410}, 
}
```

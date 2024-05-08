# Characterizing the Complexity of Social Robot Navigation Scenarios
This repository will contain the code for our paper, including a version of CrowdNav enhanced with features to enable creation of a wider range of more complex scenarios. Check back soon for the code.


## Abstract
Social robot navigation algorithms are often demonstrated in overly simplified scenarios, prohibiting the extraction of practical insights about their relevance to real-world domains. Our key insight is that an understanding of the inherent complexity of a social robot navigation scenario could help characterize the limitations of existing navigation algorithms and provide actionable directions for improvement. Through an exploration of recent literature, we identify a series of factors contributing to the Complexity of a scenario, disambiguating between contextual and robot-related ones. We then conduct a simulation study investigating how manipulations of contextual factors impact the performance of a variety of navigation algorithms. We find that dense and narrow environments correlate most strongly with performance drops, while the heterogeneity of agent policies and directionality of agents have a less pronounced effect. This motivates a shift towards developing and testing algorithms under higher Complexity settings.

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy rgl
```
2. Test policies with 500 test cases.
```
python test.py --policy rgl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy rgl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```



## Video Demo
[<img src="https://i.imgur.com/SnqVhit.png" width="70%">](https://youtu.be/U3quW30Eu3A)


## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```bibtex
@inproceedings{chen2020relational,
    title={Relational Graph Learning for Crowd Navigation},
    author={Changan Chen and Sha Hu and Payam Nikdel and Greg Mori and Manolis Savva},
    year={2020},
    booktitle={IROS}
}
@inproceedings{chen2019crowdnav,
    title={Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
    author={Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
    year={2019},
    booktitle={ICRA}
}
```

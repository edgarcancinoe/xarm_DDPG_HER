# xarm6_DDPG_HER
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm using Hindsight Experience Replay (HER) for trajectory planning of a 6-DOF UFactory xArm6 robot using OpenAI gym.

This repository contains the training code, parameter configuration, and a working model to compute inverse kinematics for a xArm6 robot to reach a target destination in cartesian coordinates.

The simulation is carried out using ```mujoco200``` and the environment ```gym-xarm6``` was provided by julio-design in the following repository: [https://github.com/julio-design/xArm6-Gym-Env](xArm6-Gym-Env). The original repository also contains a similar learning implementation, but the one here has been simplified, modularized and detailed.

The code consists of a detailed implementation of DDPG & HER with the _future_ sampling strategy, based on each algorithm's respective paper, and OpenAI [https://github.com/openai/baselines/tree/master/baselines/her](baseline's implementation of HER).

## Results

## Installation

### Requirements


## Reference
**DDPG Paper**: <a>https://arxiv.org/abs/1509.02971</a>
**Hindsight Experience Replay**: https://arxiv.org/abs/1707.01495
---
The output of this code, i.e. the actor policy to reach the target positions has been translated into in ROS for a real xArm6 in [https://github.com/edgarcancinoe/xArm6_DDPG_ROS/blob/master/README.md](this) repo.

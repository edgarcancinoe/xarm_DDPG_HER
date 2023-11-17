# xarm6_DDPG_HER
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm using Hindsight Experience Replay (HER) for trajectory planning of a 6-DOF UFactory xArm6 robot.

This repository contains the training code and configuration, and also a working model to compute inverse kinematics for a xArm6 robot to reach a target destination in cartesian coordinates.

The simulation is carried out using ```mujoco200`` and the environment 

The code consists of a detailed implementation of DDPG & HER with the _future_ sampling strategy, based on each algorithm's respective paper, and OpenAI's implementation of
HER.

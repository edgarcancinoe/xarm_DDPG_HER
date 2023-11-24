# xarm6_DDPG_HER
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm using Hindsight Experience Replay (HER) for trajectory planning of a 6-DOF robotic arm OpenAI gym.

This repository contains the training code, parameter configuration, and a working model to compute inverse kinematics for a custom robot designed by the Roborregos Robotics team at Tecnologico de Monterrey for the Robocup@home competition, to reach a target destination in cartesian coordinates. The robot's arm is a UFactory xArm6.

The simulation is carried out using ```mujoco210``` and the environment is a custom gym environment, based on the ```gym-xarm6``` environment, provided by _julio-design_ in the following repository: [xArm6-Gym-Env](https://github.com/julio-design/xArm6-Gym-Env). The original repository also contains a similar learning implementation, but a different, custom DDPG + HER implementation is used, and the overall code has been simplified, modularized and detailed.

The implementation of DDPG & HER with the _future_ sampling strategy is based on each algorithm's respective paper<sup>1, 2</sup>, and [OpenAI baseline's implementation of HER](https://github.com/openai/baselines/tree/master/baselines/her).

## Results

Training success rate            |  Actor Loss over training
:-------------------------:|:-------------------------:
![](https://github.com/edgarcancinoe/home_DDPG_HER/blob/master/success_rate.png) |  ![](https://github.com/edgarcancinoe/home_DDPG_HER/blob/master/actor_loss.png)

Test visualization
:-------------------------:
![](https://github.com/edgarcancinoe/home_DDPG_HER/blob/master/results.gif) 

## Installation
 
##### Requirements
* Python 3.6+
* gym===0.15.7
* mpi4py===3.1.4
* mujoco2.1.0
* mujoco-py===2.1.2.14
* numpy===1.26.1
* openai baselines
* torch===2.0.0

##### Install
1. Clone this repository
   ```
   git clone https://github.com/edgarcancinoe/home_DDPG_HER.git
   ```
   
3. Download MuJoCo [here](https://github.com/openai/mujoco-py#install-mujoco).
   
5. Create ```.mujoco``` folder in home directory and unzip download file there.
   
7. Modify ```.bashrc``` and add:
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/home/dir/.mujoco/mujoco200/bin
   ```
   
## Use
1. Navigate to repository folder and activate Python environment.
2. To start training of the model:
   ```
   python3 main.py -t
   ```
   A number of epochs will run (default is 24 but only about 7 are necessary). After each epoch the policy is evaluated and the model is saved in the path specified by the 'save_dir' parameter found in _main.py_ as _env_name/model.pt_.

3. To load the saved models and see the agent act in the environment based on the current policy, run:
   ```
   python3 main.py -s <path_to_your_model>
   ```

   Which might be:
   ```
   python3 main.py -s /home/edgarcancinoe/gym-rl/home_DDPG_HER/models/xArm6Reach-v1/modelo.pt
   ```
  
## Reference
1. **DDPG Paper**: <a>https://arxiv.org/abs/1509.02971</a>
2. **Hindsight Experience Replay**: https://arxiv.org/abs/1707.01495
3. **xArm6 Gym environment**: https://github.com/julio-design/xArm6-Gym-Env
4. **Baseline's HER implementation**: https://github.com/openai/baselines/tree/master/baselines/her
---
The output of this code, i.e. the actor policy to reach the target positions has been translated into in ROS for a real xArm6 in [https://github.com/edgarcancinoe/xArm6_DDPG_ROS/blob/master/README.md](this) repository.

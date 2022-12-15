# DQN With Various Rewards on SUMO-RL

This is a class project for USU CS 5640 (Reinforcement Learning Applications)
- Using Environment: [SUMO-RL](https://github.com/LucasAlegre/sumo-rl)
- Simulation: [SUMO](https://github.com/eclipse/sumo)
- Using a Deep Q Learning agent with [Pytorch](https://pytorch.org/docs/stable/index.html)

This project utilizes a DQN with dense policy and target networks to find ideal traffic signal controls.  
It utilizes the environment, default rewards, and files representing a 2-way intersection from SUMO-RL.  
There are various rewards implemented from the environment in SUMO-RL, these are tested using different objective values.

Final video presentation link: [Final Presentation](https://youtu.be/hGR2TH-PPRQ)

## Repository Contents

- dqn_learning/ 
  - python files used to generate models and plots (main.py)
  - Python file used to run gui-enabled simulation of specific generated model
- nets/ network files to run simulation
- sumo_rl/ environment files (modified from original to include some additional metrics)
- Final report PDF
- Final presentation slides in PDF format


## Observation Space

This simulation only has the option for a 2-way intersection so the observation space (```obs```) is as follows:

- ```obs[0:4]``` is a one-hot encoded vector indicating the current active green phase
- ```obs[4]``` is a binary variable indicating whether min_green seconds have already passed in the current phase
- ```obs[5:13]``` is the density of each lane: the number of vehicles in a lane divided by the lane's total capacity
- ```obs[13:21]``` is the queue amount of each lane: the number of queued vehicles in a lane divided by the lane's total capacity

## Action Space

The action space (```act```) is 4 discrete actions, each representing a different green phase:

- ```act[0]``` Green Signals: North-South straight/right turn
- ```act[1]``` Green Signals: North-South left turn
- ```act[2]``` Green Signals: East-West straight/right turn
- ```act[3]``` Green Signals: East-West left turn

# Reward Functions

There are four reward functions:
- Difference in waiting time between steps
- Average speed
- Queue
- Pressure

## Evaluation Metrics

After each iteration of training and testing an agent, that agent's policy NN and info about the episode is saved. The info that is saved includes:
- Total number of vehicles moved through the intersection
- Average speed at each step
- Vehicles in intersection at each step
- Average and total wait times at each step
- Vehicles queued at each step

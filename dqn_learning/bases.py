import gymnasium as gym
import sumo_rl
import os
from tqdm.auto import tqdm
import pandas as pd

from dqn_learning.train_periodic_test import run_to_csv

# Unoptimized base case, cycles at 30 steps through each phase
def run_bases(episode_steps):

    os.makedirs(f'bases/output', exist_ok=True)

    env = gym.make('sumo-rl-v0',
                    net_file='../nets/2way-single-intersection/single-intersection.net.xml',
                    route_file='../nets/2way-single-intersection/single-intersection-gen.rou.xml',
                    use_gui=True,
                    num_seconds=100000)


    cycles = [30, 30, 30, 30]

    rewards = [0]
    cumulative_rewards = [0]

    state, info = env.reset()
    action, start_step = 0, 0
    for step in tqdm(range(1, episode_steps+1), position=1, leave=False, desc='     steps'):
        if step - start_step > cycles[action]:
            action = (action + 1) % 4
            start_step = step
        next_state, reward, *_ = env.step(action)
        rewards.append(reward)
        cumulative_rewards.append(cumulative_rewards[-1] + reward)
        run_to_csv(env.metrics_to_df(), rewards, cumulative_rewards, 'bases', 0, 'cycle')

    info = pd.read_csv(f'bases/output/cycle_metrics_0.csv')
    total_completed_vehicles = info['system_total_completed_vehicles'].iloc[-1]
    average_speed = info['t_average_speed'].mean()
    average_queue = info['agents_total_stopped'].mean()

    print(f'Cycle Model:\n'
          f'          Total completed vehicles = {total_completed_vehicles}\n'
          f'          Average speed =            {average_speed}\n'
          f'          Average queue length =     {average_queue}')

    return total_completed_vehicles

import gymnasium as gym
import sumo_rl
import os
from tqdm.auto import tqdm
import torch

from agent import DQNAgent
from dqn_learning.train import run_to_csv


def run_test_loop(reward_func, show_gui, n_episodes, episode_steps, run_name):

    os.makedirs(f'{run_name}/output', exist_ok=True)

    env = gym.make('sumo-rl-v0',
                    net_file='../nets/2way-single-intersection/single-intersection.net.xml',
                    route_file='../nets/2way-single-intersection/single-intersection-gen.rou.xml',
                    num_seconds=100000,
                    use_gui=show_gui,
                    reward_fn=reward_func)

    for run in tqdm(range(1, n_episodes + 1), position=0, leave=False, desc='    runs'):
        agent = DQNAgent(env)
        agent.test_mode()
        agent.policy.load_state_dict(torch.load(f'{run_name}/models/modelstatedict_' + str(run) + '.pth'))
        agent.policy.to(agent.device)
        agent.policy.eval()

        rewards = [0]
        cumulative_rewards = [0]

        state, info = env.reset()
        state = torch.tensor([state], device=agent.device)

        for _ in tqdm(range(1, episode_steps+1), position=1, leave=False, desc='     steps'):
            action = agent.act(state)
            next_state, reward, *_ = env.step(action)
            state = torch.tensor([next_state], device=agent.device)

            rewards.append(reward)
            cumulative_rewards.append(cumulative_rewards[-1] + reward)

        run_to_csv(env.metrics_to_df(), rewards, cumulative_rewards, run_name, run, 'test')

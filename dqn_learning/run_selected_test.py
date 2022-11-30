from agent import DQNAgent
import warnings
import gymnasium as gym
import sumo_rl
from tqdm.auto import tqdm
import torch

warnings.filterwarnings("ignore")

reward_func = 'average-speed'
run_name = 'speed'
run = 10
episode_steps = 2000

if __name__ == '__main__':
    env = gym.make('sumo-rl-v0',
                    net_file='../nets/2way-single-intersection/single-intersection.net.xml',
                    route_file='../nets/2way-single-intersection/single-intersection-gen.rou.xml',
                    num_seconds=100000,
                    use_gui=True,
                    reward_fn=reward_func)

    agent = DQNAgent(env, f'{run_name}/models/modelstatedict_' + str(run) + '.pth')
    agent.test_mode()

    rewards = [0]
    cumulative_rewards = [0]

    state, info = env.reset()
    state = torch.tensor([state], device=agent.device)

    for _ in tqdm(range(1, episode_steps+1), position=1, leave=False, desc='     steps'):
        action = agent.act(state)
        print(action)
        next_state, reward, *_ = env.step(action)
        state = torch.tensor([next_state], device=agent.device)

        rewards.append(reward)
        cumulative_rewards.append(cumulative_rewards[-1] + reward)


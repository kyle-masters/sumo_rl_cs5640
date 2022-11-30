import gymnasium as gym
import sumo_rl
import os
from tqdm.auto import tqdm
import torch

from agent import DQNAgent, ReplayMemory


def run_to_csv(dataframe, rewards, cumulative_rewards, run_name, run, mode):
    dataframe['rewards'] = rewards
    dataframe['cumulative_rewards'] = cumulative_rewards

    os.makedirs(f'{run_name}/output', exist_ok=True)
    dataframe.to_csv(f'{run_name}/output/{mode}_metrics_{run}.csv', index=False)


def run_train_loop(reward_func, show_gui, n_episodes, episode_steps, mem_capacity, batch_size, n_batches,
                      batch_step, target_step, run_name):

    os.makedirs(f'{run_name}/models', exist_ok=True)
    os.makedirs(f'{run_name}/output', exist_ok=True)

    env = gym.make('sumo-rl-v0',
                    net_file='../nets/2way-single-intersection/single-intersection.net.xml',
                    route_file='../nets/2way-single-intersection/single-intersection-gen.rou.xml',
                    num_seconds=100000,
                    use_gui=show_gui,
                    reward_fn=reward_func)

    agent = DQNAgent(env)
    memory = ReplayMemory(mem_capacity)

    errors = []

    for run in tqdm(range(1, n_episodes + 1), position=0, leave=False, desc='    runs'):
        rewards = [0]
        cumulative_rewards = [0]

        state, info = env.reset()
        state = torch.tensor([state], device=agent.device)

        for step in tqdm(range(1, episode_steps+1), position=1, leave=False, desc='     steps'):
            action = agent.act(state)
            next_state, reward, *_ = env.step(action)

            memory.push(
                state,
                action,
                torch.tensor([next_state], device=agent.device),
                torch.tensor([reward], device=agent.device)
            )

            state = torch.tensor([next_state], device=agent.device)

            rewards.append(reward)
            cumulative_rewards.append(cumulative_rewards[-1] + reward)

            if step % batch_step == 0:
                for n in range(n_batches):
                    memory_batch = memory.sample(batch_size)
                    loss = agent.optimize_model(memory_batch)
                errors.append(loss.item())

            if step % target_step == 0:
                agent.update_target()

        torch.save(agent.policy.state_dict(), f'{run_name}/models/modelstatedict_' + str(run) + '.pth')
        torch.save(agent.policy, f'{run_name}/models/model_' + str(run) + '.pth')

        agent.update_randomness()

        run_to_csv(env.metrics_to_df(), rewards, cumulative_rewards, run_name, run, 'train')

import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_runs(run_names, n_runs, plot_names, batch_step, base_line):

    total_completed_vehs = []
    rewardss = []

    total_completed_vehs.append([base_line for i in range(n_runs)])
    rewardss.append([base_line for i in range(n_runs)])

    for i in range(len(run_names)):
        # Plot errors
        # with open(f'loss_{run_names[i]}.pkl', 'rb') as f:
        #     losses = pickle.load(f)
        # plt.plot([batch_step*i for i in range(len(losses))], losses)
        # plt.xlabel('Steps')
        # plt.ylabel('Loss')
        # plt.title(f'{plot_names[i]} Losses')
        # plt.show()

        total_completed_vehicles = 0
        average_speed = 0
        average_queue = 0

        total_completed_vehicles_plot = []
        average_speed_plot = []
        rewards = []

        for j in range(1, n_runs + 1):
            info = pd.read_csv(f'{run_names[i]}/output/test_metrics_{j}.csv')

            if j == n_runs:
                final_veh = info['system_total_completed_vehicles'].iloc[-1]

            rewards.append(info['cumulative_rewards'].iloc[-1])
            total_completed_vehicles_plot.append(info['system_total_completed_vehicles'].iloc[-1])
            average_speed_plot.append(info['t_average_speed'].mean())

            total_completed_vehicles = max(total_completed_vehicles, info['system_total_completed_vehicles'].iloc[-1])
            average_speed = max(average_speed, info['t_average_speed'].mean())
            average_queue = max(average_queue, info['agents_total_stopped'].mean())


        rewards_min = min(rewards)
        rewards_max = max(rewards)

        if i == 3:
            rewards_min = max(rewards)
            rewards_max = min(rewards)

        rewards = [(rewards[i] - rewards_min) * 100 / (rewards_max - rewards_min) for i in range(len(rewards))]

        total_completed_vehs.append(total_completed_vehicles_plot)
        rewardss.append(rewards)

        print(f'{plot_names[i]} Model {j:2}:\n'
              f'          Maximum throughput =   {total_completed_vehicles}\n'
              f'          Average speed =        {average_speed}\n'
              f'          Average queue length = {average_queue}\n'
              f'          Final throughput =     {final_veh}')

        # Generate individual plots for reward functions
        # plt.plot(list(range(1, n_runs+1)), total_completed_vehicles_plot)
        # plt.xlabel('Episode')
        # plt.ylabel('Total Completed Vehicles')
        # plt.title(f'{plot_names[i]} Vehicle Throughput')
        # plt.show()
        #
        # plt.plot(list(range(1, n_runs+1)), average_speed_plot)
        # plt.xlabel('Episode')
        # plt.ylabel('Vehicle Speed')
        # plt.title(f'{plot_names[i]} Average Vehicle Speed')
        # plt.show()


    names = ['Baseline', 'Wait Time', 'Speed', 'Queue Length', 'Vehicle Count']

    # Generate PRESENTATION plots
    # for i in range(3):
    #     if i == 0:
    #         plt.plot(list(range(1, n_runs+1)), rewardss[i])
    #     else:
    #         plt.plot(list(range(1, n_runs+1)), rewardss[i], label=names[i])
    #
    # plt.legend()
    # plt.xlabel('Episode')
    # plt.ylabel('Adjusted Award')
    # plt.title(f'Reward by Reward Function')
    # plt.ylim(0, 100)
    # plt.show()
    #
    # for i in [0, 3, 4]:
    #     if i == 0:
    #         plt.plot(list(range(1, n_runs+1)), rewardss[i])
    #     else:
    #         plt.plot(list(range(1, n_runs+1)), rewardss[i], label=names[i])
    #
    # plt.legend()
    # plt.xlabel('Episode')
    # plt.ylabel('Adjusted Award')
    # plt.title(f'Reward by Reward Function')
    # plt.ylim(0, 100)
    # plt.show()
    #
    # for i in range(3):
    #     plt.plot(list(range(1, n_runs+1)), total_completed_vehs[i], label=names[i])
    #
    # plt.legend()
    # plt.xlabel('Episode')
    # plt.ylabel('Total Completed Vehicles')
    # plt.title(f'Vehicle Throughput by Reward')
    # plt.ylim(0, 700)
    # plt.show()
    #
    # for i in [0, 3, 4]:
    #     plt.plot(list(range(1, n_runs+1)), total_completed_vehs[i], label=names[i])
    #
    # plt.legend()
    # plt.xlabel('Episode')
    # plt.ylabel('Total Completed Vehicles')
    # plt.title(f'Vehicle Throughput by Reward')
    # plt.ylim(0, 700)
    # plt.show()

    # Generate final plots
    for i in range(1, len(names)):
        plt.plot(list(range(1, n_runs+1)), total_completed_vehs[0], label=names[0])
        plt.plot(list(range(1, n_runs+1)), total_completed_vehs[i], label=names[i])

        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Total Completed Vehicles')
        plt.title(f'Vehicle Throughput: {names[i]}')
        plt.ylim(0, 700)
        plt.show()

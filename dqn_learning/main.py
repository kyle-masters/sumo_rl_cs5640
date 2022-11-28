import warnings
from dqn_learning import train, test
import time

warnings.filterwarnings("ignore")

# Hyper parameters
n_episodes = 10
train_steps = 20000
mem_capacity = 20000
batch_size = 50
n_batches = 8
batch_step = 500
target_step = 2000

test_steps = 5000


def print_heading(to_print, level=1, indent=1):
    space = '  '
    if level == 0:
        sep_char = '='
    elif level == 1:
        sep_char = '-'
    elif level == 2:
        sep_char = '~'
    else:
        sep_char = ''

    if level == 0 or level == 1 or level == 2:
        to_print = f'{(sep_char * len(to_print))}\n' \
                   f'{space * indent}{to_print}\n' \
                   f'{space * indent}{(sep_char * len(to_print))}\n'

    print(f'{space * indent}{to_print}')


if __name__ == '__main__':
    reward_funcs = ['diff-waiting-time', 'average-speed', 'queue', 'pressure']
    run_names = ['wait', 'speed', 'queue', 'pressure']

    start_time = time.time()

    print_str = f'Beginning experiments: {len(reward_funcs)} functions, {n_episodes} episodes, {train_steps} training steps, {test_steps} testing steps'
    print_heading(print_str, 0, 0)
    for i, reward_func in enumerate(reward_funcs):
        print_str = f'Experiment {i+1}: {reward_func}'
        print_heading(print_str, 1, 1)

        print_heading('Beginning training loop...', -1, 1)
        sub_start = time.time()
        train.run_train_loop(reward_func, False, n_episodes, train_steps, mem_capacity, batch_size, n_batches,
                             batch_step, target_step, run_names[i])
        print(f'  Loop time: {time.time() - sub_start:.2f} seconds, elapsed time: {time.time() - start_time:.2f} seconds\n')

        print_heading('Beginning testing loop...', -1, 1)
        sub_start = time.time()
        test.run_test_loop(reward_func, False, n_episodes, test_steps, run_names[i])
        print(f'  Loop time: {time.time() - sub_start:.2f} seconds, elapsed time: {time.time() - start_time:.2f} seconds\n')

    print(f'All experiments complete: total time {time.time() - start_time:.2f} seconds')

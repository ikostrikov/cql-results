import argparse

import yaml

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=5,
    help='number of random seeds to generate')
parser.add_argument(
    '--env-names',
    # default="hopper-expert-v0;hopper-medium-expert-v0;hopper-medium-v0;hopper-random-v0;hopper-medium-replay-v0",
    # default="halfcheetah-expert-v0;halfcheetah-medium-expert-v0;halfcheetah-medium-v0;halfcheetah-random-v0;halfcheetah-medium-replay-v0",
    default="walker2d-expert-v0;walker2d-medium-expert-v0;walker2d-medium-v0;walker2d-random-v0;walker2d-medium-replay-v0",
    help='environment name separated by semicolons')
args = parser.parse_args()

d4rl_template = "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python examples/cql_mujoco_new.py --env={0} --policy_lr=1e-4 --seed={1} --lagrange_thresh=-1.0 --gpu={2} --min_q_weight=5.0 --min_q_version=3"

template = d4rl_template

config = {"session_name": "run-all", "windows": []}

gpu_id = 0
for i in range(args.num_seeds):
    panes_list = []
    for env_name in args.env_names.split(';'):
        panes_list.append(
            template.format(env_name, i, gpu_id % 4))
        gpu_id += 1

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "shell_command_before": "conda activate cql",
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
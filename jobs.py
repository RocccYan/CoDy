import itertools
import os


dataset = 'dblp.v13'
batch_size = 256
sampler = 'hgt'
# check
model_name = 'hgt'
num_epochs = 2
hidden_channels_op =  [32] #[32, 128, 300]
num_heads_op = [1] #[2,4,8]
num_layers_op = [1] #[1,2,4]

seed_op = [0] #[0,1,42]
num_samples_op = [10]
lr_op = [0.001] # [0.01,0.001,0.0005]

# run jobs
# num_epochs = 100
# hidden_channels_op =  [32, 64] #[32, 128, 300]
# num_heads_op = [4] #[2,4,8]
# num_layers_op = [4] #[1,2,4]

# seed_op = [0,1] #[0,1,42]
# num_samples_op = [10,50,100,200]
# lr_op = [0.001,0.0005] # [0.01,0.001,0.0005]


for seed, num_samples, lr, hidden_channels, num_heads, num_layers in itertools.product(
    seed_op,num_samples_op,lr_op,hidden_channels_op,num_heads_op,num_layers_op):
    # task_name = f"mm_{dataset_name}_{models_dir_graph}_{combine_mode}_scales"
    os.system(
        f"python main.py "
        f"--dataset {dataset} "
        f"--model_name {model_name} "
        f"--hidden_channels {hidden_channels} "
        f"--num_heads {num_heads} "
        f"--num_layers {num_layers} "
        f"--seed {seed} "
        f"--sampler {sampler} "
        f"--num_samples {num_samples} "
        f"--lr {lr} "
        f"--num_epochs {num_epochs} "
        f"--batch_size {batch_size} "
    )


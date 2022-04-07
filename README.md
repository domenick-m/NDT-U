# NDT-U
Neural Data Transformer with Undivided Attention
<p align="center">
    <img width="95%" src="assets/teaser.png" />
</p>

This is the code for my submission to the [Neural Latents Benchmark Challenge 2021, Phase 2](https://neurallatents.github.io/). My model is a simplified version of Joel Ye's [neural-data-transformers](https://github.com/snel-repo/neural-data-transformers) with changes to the MultiheadedSelfAttention. Specfically, the dimensions of the Keys, Querys, and Values were changed from (n_neurons / n_heads) to n_neurons. This allows each head to see the activity of all neurons, whereas before each head had to construct a full representation of the activity in a timestep with only (n_neurons / n_heads) dimensions.

# Setup
First start by recreating the environemt using `conda/miniconda`. This can be achieved by running: `conda env create -f environment.yml`. Next, install [nlb_tools](https://github.com/neurallatents/nlb_tools). 


This project uses wandb to track the runs and manage the sweeps. You can turn off wandb using: `default_config.wandb.log=False`, note that sweeps will no longer work. If you wish to use wandb (reccomended), then first sign up for an account at: https://wandb.ai/site. Then run the command `wandb login` and paste your API Key.

This project was created in Linux, certain system commands may not work on other platforms.

# Usage
To train a model simply run: `python train.py`, the system will then download and extract the selected dataset. 


By default the model will be trained with the same configurations as the submissons (currently only `mc_maze_small`, `mc_maze_medium`, `mc_maze_large`, and `area2_bump`) unless the `--default` flag is used, then it will only use the configs in `default_config.py`. You may also set configs for the current run by using the configs name you want to change as a flag: `python train.py --CONFIG_TO_SET value_to_change_to`. Example: `python train.py --dataset mc_maze --epochs 1000`


To start a sweep, run: `python train.py --sweep`. To add an agent to an already running sweep, run: `python train.py --add`. To see the full list of possible arguments, run: `python train.py -h`.

To create a submission.h5 file run: `python test.py /path/to/model.pt`.

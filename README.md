# Curious Minded Machine - BusyBox Simulation Environment

## Installation

TODO

## Usage

### Generating Datasets 

#### Random Exploration

To generate random datasets use the module ```gen.generate_policy_data``` with the following arguments:

Argument | Type | Description | Default
--- | --- | --- | ---
```--viz``` | bool | True if want to visualize sim | False
```--debug``` | bool | True if want to visualize verbose output and helpful sim visualizations | False
```--n-bbs``` | int | number of Busyboxes in generated dataset | 5
```--n-samples``` | bool | number of samples per Busybox in dataset | 1
```--mech-types``` | list of strings in ['slider', 'door'] | mechanism types in dataset | 'slider'
```--fname``` | string | file path to save dataset to | does not save file if not specified
```--random-policies``` | bool | if True, generate random policies from any policy class, else only generate policies from the correct policy class | False 
```--randomness``` | int in [0,1] | if 0 then only generate correct policies for each mechanism, if 1 then generate random policies for each mechanism | 1
```--goal-config``` | float in [0.0,1.0] or None | if specified, the goal config to attempt in the dataset, else (None) goal config is randomly sampled | None
```--bb-fname``` | string | if specified, the file path of the results dataset with the desired Busyboxes for this generated dataset, else (None) random Busyboxes are generated for this dataset | None
```--no-gripper``` | bool | if True, then apply forces directly to mechanism handles, else use and apply forces to the gripper | False

#### GP-UCB Exploration 

 To generate datasets using the GP-UCB method use the module ```learning.gp.explore_single_bb``` with the following arguments:
 
Argument | Type | Description | Default
--- | --- | --- | ---
```--L``` | int | number of Busyboxes in generated dataset | required
```--M``` | int | number of interactions per Busybox | required
```--fname``` | string | file path to save dataset to | does not save file if not specified
```--bb-fname``` | string | if specified, the file path of the results dataset with the desired Busyboxes for this generated dataset, else (None) random Busyboxes are generated for this dataset | None
```--mech-types``` | list of strings in ['slider', 'door'] | mechanism types in dataset | 'slider'
```--plot``` | bool | if True then visualize GP plots during interaction | False
```--n-gp-samples``` | int | the number of samples to use when initializing an optimization seed | 500

#### Systematic

TODO

#### Active Exploration (Oudeyer Method)

TODO

### Training

To train models use the module ```learning.train``` with the following arguments:

Argument | Type | Description | Default
--- | --- | --- | ---
```--batch-size``` | int | batch size during training | required
```--hdim``` | int | number of hidden units and feature points during training | required
```--n-epochs``` | int | number of epochs to train for | 10
```--val-freq``` | int | frequency to checkand output validation error (epoch with smallest validation error is saved) | 5
```--use-cuda``` | bool | if True use CUDA tensor types, else use CPU tensors | False
```--data-fname``` | string | file path to dataset for training | required
```--save-dir``` | string | directory to save training results to | required
```--L-min``` | int | minimum number of Busyboxes to include in a trained model | 10
```--L-max``` | int | maximum number of Busyboxes to include in a trained model | 100
```--L-step``` | int | increment between L-min and L-max to train models | 10
```--M``` | int | number of interactions per Busybox to use from dataset to train models | 100
```--image-encoder``` | string in ['spatial', 'cnn'] | type of network to use when encoding images | 'spatial

### Evaluation using GP-UCB (generating regret results files)

To evaluate models use the module ```learning.gp.evaluate_models``` with the following arguments:

Argument | Type | Description | Default
--- | --- | --- | ---
```--T``` | int | number of Busyboxes to average over when calculating regrets | required
```--N``` | int | number of interactions per Busybox to learn residual reward function | required
```--models-path``` | string | path to model files. **ALL files ending .pt in this directory will be evaluated** | required
```--Ls``` | list of 3 ints | [min, max, step] of Ls to evaluate (used when searching for correct model files in models-path) | required
```---type``` | string | used to identify these results for regret plotting (eg. random, random_doors, gpucb_sliders, gpucb, etc...). **the string must contain a substring in [random, gpucb, systematic, or active] to select the line plotting color later)**| required
```--hdim``` | int | number of hidden units and feature points in given model (needed to load pyTorch model) | 16
```--bb-fname``` | string | if specified, the file path of the results dataset with the desired Busyboxes for evaluation, else (None) random Busyboxes are generated for this dataset | None
```--plot``` | bool | if True then save visualizations of reward function polar plots, GP samples, and optimization results to ```gp_plots/``` during interaction **(WARNING: this slows down the evaluation quite a bit)**| False
```--n-gp-samples``` | int | the number of samples to use when initializing an optimization seed | 500

### Plotting Regret Results

To generate regret plots use the module ```learning.make_regret_plots2``` with the following arguments:

Argument | Type | Description | Default
--- | --- | --- | ---
```--types``` | list of strings | list of types to plot (should match the ```--type``` arg used when evaluating models) | required
```--results-path``` | string | file path to search for regret results files | required

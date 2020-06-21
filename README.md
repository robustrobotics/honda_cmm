# Curious Minded Machine - BusyBox Simulation Environment

## Installation

TODO

## Usage

### To-Do List 

Generate a dataset with 100 sliders, 100 interactions each:

```buildoutcfg
python3 -m learning.gp.explore_single_bb --L 100 --M 100 --mech-types slider --fname train100.pickle
/./mc cp honda_cmm/train100.pickle honda_cmm/continual/gptrain100
```

Train a model using Contextual Prior Prediction (CPP)

```buildoutcfg
/./mc cp honda_cmm/continual/gptrain100 dataset.pickle
python3 -m learning.train --batch-size 16 --hdim 16 --data-fname dataset.pickle --save-dir cpp_models --n-epochs 50 --L-min 5 --L-max 50 --L-step 5
/./mc cp -r cpp_models/ honda_cmm/continual/test_train_models/cpp_models
```

Train a model using continual learning on pre-generated dataset
- Consider modifying batch and buffer size in `replay_train.py`
- Test the following commands on `test20.pickle` first

```buildoutcfg
/./mc cp honda_cmm/continual/gptrain100 dataset.pickle
python3 -m learning.replay_train --batch-size 16 --hdim 16 --data-fname dataset.pickle --save-dir precont_models --n-epochs 50 --L-min 100 --L-max 100
/./mc cp -r precont_models/ honda_cmm/continual/test_train_models/precont_models
```

Train a model using continual learning with continual exploration
- Use same batch and buffer size as above
- Comment out lines 487-489 of `explore_single_bb.py` and uncomment line 485 to prevent all generated data from being saved
- Also test on `test20.pickle`
```buildoutcfg
python3 -m learning.cont_train --batch-size 16 --hdim 16 --save-dir cstream_models --n-epochs 50 --L-min 5 --L-max 5
/./mc cp -r cstream_models/ honda_cmm/continual/test_train_models/cstream_models
```

Evaluate each of the models 
```buildoutcfg
/./mc cp -r honda_cmm/continual/test_train_models/cpp_models cppmodels/
python3 -m learning.gp.evaluate_models --T 50 --N 50 --models-path cppmodels --Ls 5 50 5 --type gpucb --hdim 16 --eval-method noT
```

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
```--bb-fname``` | string | if specified, the file path of the results dataset with the desired Busyboxes for this generated dataset, else (None) random Busyboxes are generated for this dataset | None

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
```--random-policies``` | bool | if True, generate random policies from any policy class, else only generate policies from the correct policy class | False 

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
```--eval-method``` | string | `T` or `noT` | None

### Plotting Regret Results

To generate regret plots use the module ```learning.make_regret_plots2``` with the following arguments:

Argument | Type | Description | Default
--- | --- | --- | ---
```--types``` | list of strings | list of types to plot (should match the ```--type``` arg used when evaluating models) | required
```--results-path``` | string | file path to search for regret results files | required
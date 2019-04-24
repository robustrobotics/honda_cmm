# Curious Minded Machine

## BusyBox Simulation Environment

### Installation

First, set-up a virtualenv with the required dependencies:
```
virtualenv --python=/usr/bin/python3 .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

1. Generate trajectories by running:
python3 actuate_model.py --model-name simple_busybox --log-name
full_busybox.json --joint-name all --duration 5

2. Fit the articulated models:
a) Change honda.launch to have the correct json log name.
b) roslaunch cmm_articulated honda.launch

3. Visualize the results by running the first part again:
python3 actuate_model.py --model-name simple_busybox --visualize --duration 10

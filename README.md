# Curious Minded Machine

## BusyBox Simulation Environment

### Installation

First, set-up a virtualenv with the required dependencies:
```
virtualenv --python=/usr/bin/python3 .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### Manipulate BusyBox with Gripper
Generate a random BusyBox and actuate it with model-based policies:

```python3 -m test_policy```

Below are some optional command line arguments:

  1. ```--viz``` runs the pyBullet GUI
  2. ```--random``` generates random policies to execute on the different Mechanisms (if not specified then the correct policies are used)
  3. ```--debug``` launches the pdb debugger and displays helpful visualizations and output (enter 'c' to continue without the debugger)
  4. ```--max-mech``` the maximum number of mechanisms to attempt to fit onto the BusyBox

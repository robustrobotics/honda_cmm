./mc config host add csail $CSAIL_ENDPOINT $CSAIL_ACCESS $CSAIL_SECRET
git clone https://${GITHUB_USERNAME}:${GITHUB_KEY}@github.com/robustrobotics/honda_cmm.git -b ${BRANCH:master} --single-branch
cd honda_cmm
/./mc cp csail/carism/active_50000.pickle tmp/active
/./mc cp csail/carism/random_50000.pickle tmp/random
mkdir runs
mkdir torch_models
/./mc rm -r --force csail/carism/runs_running
/./mc rm -r --force csail/carism/torch_models_running

#/./mc mirror -w runs csail/carism/runs_test > /dev/null 2>&1
python3 -m learning.train --batch-size 16 --hdim 16 --n-epochs 100 --use-cuda --random-data-path tmp/random --active-data-path tmp/active --mode ntrain --n-train 50000 --step 1000
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
/./mc cp -r runs csail/carism/runs_$TIMESTAMP
/./mc cp -r torch_models csail/carism/torch_models_$TIMESTAMP

# run in terminal to mirror while training
# nohup /./mc mirror -w --overwrite --remove honda_cmm/runs csail/carism/runs_running
# nohup /./mc mirror -w --overwrite --remove honda_cmm/torch_models csail/carism/torch_models_running
# doesn't seem to work for run data...

# have tried both of these options for visualizing tensorbaord while running
# 1) doesn't work. get Access Denied error when try to mirror
# 2) doesn't work. get Request failed error

# Option 1
# run in local machine terminal to visualize in tensorboard (try to connect straight to cloud)
# mc mirror -w --overwrite --remove csail/carism/runs_running runs_running
# mc mirror -w --overwrite --remove csail/carism/torch_models_running torch_models_running
# tensboard --logdir=runs_running

# Option 2 (see ~/.bash_profile for setup)
# tensboard --logdir=s3://csail/carism/runs_running

# working option:
# copy over: mc cp -r csail/carism/runs_running runs_running
# tensorboard --logdir=runs_running

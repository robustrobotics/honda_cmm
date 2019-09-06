./mc config host add csail $CSAIL_ENDPOINT $CSAIL_ACCESS $CSAIL_SECRET
git clone https://${GITHUB_USERNAME}:${GITHUB_KEY}@github.com/robustrobotics/honda_cmm.git -b ${BRANCH:master} --single-branch
cd honda_cmm
/./mc cp csail/carism/expert_rand8_50000.pickle tmp/active
/./mc cp csail/carism/random_50000.pickle tmp/random
mkdir runs
mkdir torch_models
/./mc rm -r --force csail/carism/runs_running
/./mc rm -r --force csail/carism/torch_models_running

#/./mc mirror -w runs csail/carism/runs_test > /dev/null 2>&1
python3 -m learning.train --batch-size 16 --hdim 16 --n-epochs 100 --use-cuda --random-data-path tmp/random --active-data-path tmp/active --mode ntrain
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
/./mc cp -r runs/ csail/carism/runs_expert_rand8_1_50k
/./mc cp -r torch_models/ csail/carism/torch_models_expert_rand8_1_50k

# run in terminal to mirror while training (doesn't work)
# nohup /./mc mirror -w --overwrite --remove honda_cmm/runs csail/carism/runs_running
# nohup /./mc mirror -w --overwrite --remove honda_cmm/torch_models csail/carism/torch_models_running

# have tried both of these options for visualizing tensorbaord while running
# 1) doesn't work. get Access Denied error when try to mirror
# 2) doesn't work. get Request failed error

# Option 1
# run in local machine terminal to visualize in tensorboard (try to connect straight to cloud)
# mc mirror -w --overwrite --remove csail/carism/runs_running runs_running
# mc mirror -w --overwrite --remove csail/carism/torch_models_running torch_models_running
# tensboard --logdir=runs_running

# Option 2 (see ~/.bash_profile for setup)
# tensorboard --logdir=s3://carism/runs_running

# working option:
# ./mc rm -r --force csail/carism/runs_running
# copy from pod to s3: ./mc cp -r honda_cmm/runs/ csail/carism/runs_running
# copy over to local machine: mc cp -r csail/carism/runs_running .
# tensorboard --logdir=runs_running

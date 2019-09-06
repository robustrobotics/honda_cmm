./mc config host add csail $CSAIL_ENDPOINT $CSAIL_ACCESS $CSAIL_SECRET
git clone https://${GITHUB_USERNAME}:${GITHUB_KEY}@github.com/robustrobotics/honda_cmm.git -b ${BRANCH:master} --single-branch
cd honda_cmm
export OMP_NUM_THREADS=1

/./mc rm -r --force csail/carism/runs_active
/./mc rm -r --force csail/carism/csail/carism/torch_models_prior

python3 -m learning.train_active --batch-size 16 --hdim 16 --n-epochs 200 --n-bbs 50 --use-cuda --n-inter 300 --n-prior 100 --lite

/./mc rm -r --force csail/carism/runs_active_cluster
/./mc rm -r --force csail/carism/csail/carism/torch_models_prior_cluster
/./mc cp -r runs_active/ csail/carism/runs_active_cluster
/./mc cp -r torch_models_prior/ csail/carism/torch_models_prior_cluster

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

./mc config host add csail $CSAIL_ENDPOINT $CSAIL_ACCESS $CSAIL_SECRET
git clone https://${GITHUB_USERNAME}:${GITHUB_KEY}@github.com/robustrobotics/honda_cmm.git -b ${BRANCH:master} --single-branch
cd honda_cmm
/./mc cp csail/carism/active_100000.pickle tmp/data1
/./mc cp csail/carism/random_100000.pickle tmp/data2
tensorboard --logdir=runs &
python3 -m learning.train --batch-size 8 --hdim 16 --n-epochs 200 --use-cuda --data-path tmp/data1 --data-path2 tmp/data2 --mode ntrain --n-train 100000
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
/./mc cp -r runs csail/carism/runs_$TIMESTAMP

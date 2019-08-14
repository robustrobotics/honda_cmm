./mc config host add csail $CSAIL_ENDPOINT $CSAIL_ACCESS $CSAIL_SECRET
git clone https://${GITHUB_USERNAME}:${GITHUB_KEY}@github.com/robustrobotics/honda_cmm.git -b ${BRANCH:master} --single-branch
cd honda_cmm
/./mc cp csail/carism/small_set tmp/data1
/./mc cp csail/carism/small_set tmp/data2
python3 -m learning.train --batch-size 5 --hdim 5 --n-epochs 10 --use-cuda \
        --data-path tmp/data1 --data-path2 tmp/data2 --mode ntrain --n-train 100

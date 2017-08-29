# Private version of Caffe
# export CAFFE_HOME=${HOME}/colorcaffe
# export PATH=${CAFFE_HOME}/bin:$PATH
# export PYTHONPATH=${CAFFE_HOME}/python:$PYTHONPATH
# export PYTHONPATH=${CAFFE_HOME}/python/caffe/proto:$PYTHONPATH
# export LD_LIBRARY_PATH=${CAFFE_HOME}/lib:$LD_LIBRARY_PATH

for RANDSEED in $(seq 2 8)
do

exec 5<zoo/rotation_list.csv
while IFS=, read -u5 NAME LAYER COLORDEPTH DSIZE PBATCH TDEPTH
do

script/rundissect.sh \
  --model "${NAME}_r${RANDSEED}" \
  --rotation_seed ${RANDSEED} \
  --proto zoo/$NAME.prototxt \
  --weights zoo/$NAME.caffemodel \
  --probebatch $PBATCH \
  --tallydepth $TDEPTH \
  --colordepth $COLORDEPTH \
  --resolution $DSIZE \
  --mean "$MEAN" \
  --layers "$LAYER" \
  --force none \
  --endafter none \
  --dataset dataset/broden1_$DSIZE

done

done


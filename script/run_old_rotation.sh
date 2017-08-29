# Private version of Caffe
# export CAFFE_HOME=${HOME}/colorcaffe
# export PATH=${CAFFE_HOME}/bin:$PATH
# export PYTHONPATH=${CAFFE_HOME}/python:$PYTHONPATH
# export PYTHONPATH=${CAFFE_HOME}/python/caffe/proto:$PYTHONPATH
# export LD_LIBRARY_PATH=${CAFFE_HOME}/lib:$LD_LIBRARY_PATH

for TENALPHA in $(seq 1 10)
do

ALPHA=$(echo "scale=2; ${TENALPHA}/10" | bc)
RANDSEED=1

exec 5<zoo/old_rotation.csv
while IFS=, read -u5 NAME LAYER COLORDEPTH DSIZE PBATCH TDEPTH
do

script/rundissect.sh \
  --model "rotation_$(printf %02d ${TENALPHA})0" \
  --rotation_seed ${RANDSEED} \
  --rotation_power ${ALPHA} \
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


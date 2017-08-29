# Private version of Caffe
export CAFFE_HOME=${HOME}/customcaffe/caffe_private_pascal/distribute
export PATH=${CAFFE_HOME}/bin:$PATH
export PYTHONPATH=${CAFFE_HOME}/python:$PYTHONPATH
export PYTHONPATH=${CAFFE_HOME}/python/caffe/proto:$PYTHONPATH
export LD_LIBRARY_PATH=${CAFFE_HOME}/lib:$LD_LIBRARY_PATH

exec 5<zoo/list_all.csv
while IFS=, read -u5 NAME LAYER COLORDEPTH DSIZE PBATCH TDEPTH
do

script/rundissect.sh \
  --model $NAME \
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

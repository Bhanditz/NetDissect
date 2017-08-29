#!/usr/bin/env bash

# To use this, put the caffe model to be tested in the "zoo" directory
# the following naming convention for a model called "vgg16_places365":
#
# zoo/caffe_reference_places365.caffemodel
# zoo/caffe_reference_places365.prototxt
#
# and then, with scipy and pycaffe available in your python, run:
#
# ./truncdissect.sh --trunc 10000 \
#      --model caffe_reference_places365 --layers "conv4 conv5"
#
# the output will be placed in a directory dissection/caffe_reference_places365/
#
# More options are listed below.

# Defaults
THRESHOLD="0.04"
WORKDIR="dissection"
TALLYDEPTH=2048
PARALLEL=4
TALLYBATCH=16
PROBEBATCH=64
QUANTILE="0.005"
COLORDEPTH="3"
CENTERED="c"
MEAN="0 0 0"
FORCE="none"
ENDAFTER="none"
MODELDIR="zoo"
ROTATION_SEED=""
ROTATION_POWER="1"

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Parse command-line arguments. http://stackoverflow.com/questions/192249

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -t|--trunc)
    TRUNC="$2"
    shift
    ;;
    -d|--model)
    DIR="$2"
    shift
    ;;
    -w|--weights)
    WEIGHTS="$2"
    shift
    ;;
    -p|--proto)
    PROTO="$2"
    shift
    ;;
    -l|--layers)
    LAYERS="$2"
    shift
    ;;
    -s|--dataset)
    DATASET="$2"
    shift
    ;;
    --colordepth)
    COLORDEPTH="$2"
    shift
    ;;
    --resolution)
    RESOLUTION="$2"
    shift
    ;;
    --probebatch)
    PROBEBATCH="$2"
    shift
    ;;
    -t|--threshold)
    THRESHOLD="$2"
    shift
    ;;
    --tallydepth)
    TALLYDEPTH="$2"
    shift
    ;;
    --tallybatch)
    TALLYBATCH="$2"
    shift
    ;;
    --mean)
    MEAN="$2"
    shift
    ;;
    --rotation_seed)
    ROTATION_SEED="$2"
    shift
    ;;
    --rotation_power)
    ROTATION_POWER="$2"
    shift
    ;;
    -w|--workdir)
    WORKDIR="$2"
    shift
    ;;
    -f|--force)
    FORCE="$2"
    shift
    ;;
    --endafter)
    ENDAFTER="$2"
    shift
    ;;
    --gpu)
    export CUDA_VISIBLE_DEVICES="$2"
    shift
    ;;
    *)
    echo "Unknown option" $key
    exit 3
    # unknown option
    ;;
esac
shift # past argument or value
done

# Get rid of slashes in layer names for directory purposes
LAYERA=(${LAYERS//\//-})

# For expanding globs http://stackoverflow.com/questions/2937407
function globexists {
  set +f
  test -e "$1" -o -L "$1";set -f
}

if [ -z $DIR ]; then
  echo '--model directory' must be specified
  exit 1
fi

if [ -z "${LAYERS}" ]; then
  echo '--layers layers' must be specified
  exit 1
fi

# Set up directory to work in, and lay down pid file etc.
mkdir -p $WORKDIR/$DIR
if [ -z "${FORCE##*pid*}" ] || [ ! -e $WORKDIR/$DIR/job.pid ]
then
    exec &> >(tee -a "$WORKDIR/$DIR/job.log")
    echo "Beginning pid $$ on host $(hostname) at $(date)"
    trap "rm -rf $WORKDIR/$DIR/job.pid" EXIT
    echo $(hostname) $$ > $WORKDIR/$DIR/job.pid
else
    echo "Already running $DIR at $(cat $WORKDIR/$DIR/job.pid)"
    exit 1
fi

if [ "$COLORDEPTH" -le 0 ]
then
  (( COLORDEPTH = -COLORDEPTH ))
  CENTERED=""
fi

if [ -z "${CENTERED##*c*}" ]
then
  MEAN="109.5388 118.6897 124.6901"
fi

# Convention: dir, weights, and proto all have the same name
if [[ -z "${WEIGHTS}" && -z "${PROTO}" ]]
then
  WEIGHTS="zoo/$DIR.caffemodel"
  PROTO="zoo/$DIR.prototxt"
fi

echo DIR = "${DIR}"
echo LAYERS = "${LAYERS}"
echo DATASET = "${DATASET}"
echo COLORDEPTH = "${COLORDEPTH}"
echo RESOLUTION = "${RESOLUTION}"
echo WORKDIR = "${WORKDIR}"
echo WEIGHTS = "${WEIGHTS}"
echo PROTO = "${PROTO}"
echo THRESHOLD = "${THRESHOLD}"
echo PROBEBATCH = "${PROBEBATCH}"
echo TALLYDEPTH = "${TALLYDEPTH}"
echo TALLYBATCH = "${TALLYBATCH}"
echo MEAN = "${MEAN}"
echo FORCE = "${FORCE}"
echo ENDAFTER = "${ENDAFTER}"

# Set up rotation flag if rotation is selected
ROTATION_FLAG=""
if [ ! -z "${ROTATION_SEED}" ]
then
    ROTATION_FLAG=" --rotation_seed ${ROTATION_SEED}
                    --rotation_unpermute 1
                    --rotation_power ${ROTATION_POWER} "
fi

# Step 5: we just run over the tally file to extract whatever score we
# want to derive.  That gets summarized in a [layer]-result.csv file.
if [ -z "${FORCE##*result*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/%s-t$TRUNC-result.csv" "${LAYERA[@]}" )
then

echo 'Generating result.csv'
python src/makeresult.py \
    --trunc $TRUNC \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS

[[ $? -ne 0 ]] && exit $?

echo makeresult > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*result*}" ]
then
  exit 0
fi

# Step 6: now generate the HTML visualization and images.
if [ -z "${FORCE##*report*}" ] || \
  ! ls $(printf " $WORKDIR/$DIR/html/%s-t$TRUNC.html" "${LAYERA[@]}") || \
  ! ls $(printf " $WORKDIR/$DIR/html/image/%s-t$TRUNC-bargraph.svg" "${LAYERA[@]}")
then

echo 'Generating report'
python src/report.py \
    --trunc $TRUNC \
    --directory $WORKDIR/$DIR \
    --blobs $LAYERS \
    --threshold ${THRESHOLD}

[[ $? -ne 0 ]] && exit $?

echo viewprobe > $WORKDIR/$DIR/job.done
fi

if [ -z "${ENDAFTER##*view*}" ]
then
  exit 0
fi


echo finished > $WORKDIR/$DIR/job.done

if [ -z "${ENDAFTER##*graph*}" ]
then
  exit 0
fi


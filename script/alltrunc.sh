#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

script/rundissect.sh \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 500 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 1000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 2000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 4000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 8000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 16000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"
script/truncdissect.sh --trunc 32000 \
    --model caffe_reference_places365 \
    --layers "conv1 conv2 conv3 conv4 conv5"

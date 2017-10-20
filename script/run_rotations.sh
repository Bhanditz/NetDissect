#!/usr/bin/env bash

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_000 \
    --rotation_power 0.0 \

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_020 \
    --rotation_power 0.2 \

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_040 \
    --rotation_power 0.4 \

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_060 \
    --rotation_power 0.6 \

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_080 \
    --rotation_power 0.8 \

script/rundissect.sh \
    --dataset dataset/broden1_227 \
    --resolution 227 \
    --proto zoo/caffe_reference_places205.prototxt \
    --weights zoo/caffe_reference_places205.caffemodel \
    --layers conv5 \
    --rotation_seed 1 \
    --model uprot_100 \
    --rotation_power 1.0 \

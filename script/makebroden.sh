#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Make the sourcedata directory if it does not exist
if [ ! -d sourcedata ]
then
  mkdir sourcedata
fi

# PASCAL 2010 Images
if [ ! -f sourcedata/pascal/VOC2010/ImageSets/Segmentation/train.txt ]
then

echo "Downloading Pascal VOC2010 images"
mkdir -p sourcedata/pascal
pushd sourcedata/pascal
wget --progress=bar \
   http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar \
   -O VOCtrainval_03-May-2010.tar
tar xvf VOCtrainval_03-May-2010.tar
rm VOCtrainval_03-May-2010.tar
mv VOCdevkit/* .
rmdir VOCdevkit
popd

fi


# PASCAL Part sourcedata
if [ ! -f sourcedata/pascal/part/part2ind.m ]
then

echo "Downloading Pascal Part Dataset"
mkdir -p sourcedata/pascal/part
pushd sourcedata/pascal/part
wget --progress=bar \
   http://www.stat.ucla.edu/~xianjie.chen/pascal_part_sourcedata/trainval.tar.gz \
   -O trainval.tar.gz
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# PASCAL Context sourcedata
if [ ! -f sourcedata/pascal/context/labels.txt ]
then

echo "Downloading Pascal Context Dataset"
mkdir -p sourcedata/pascal/context
pushd sourcedata/pascal/context
wget --progress=bar \
   http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz \
   -O trainval.tar.gz
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# DTD
if [ ! -f sourcedata/dtd/dtd-r1.0.1/imdb/imdb.mat ]
then

echo "Downloading Describable Textures Dataset"
mkdir -p sourcedata/dtd
pushd sourcedata/dtd
wget --progress=bar \
   https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz \
   -O dtd-r1.0.1.tar.gz
tar xvzf dtd-r1.0.1.tar.gz
mv dtd dtd-r1.0.1
rm dtd-r1.0.1.tar.gz
popd

fi


# OpenSurfaces
if [ ! -f sourcedata/opensurfaces/photos.csv ]
then

echo "Downloading OpenSurfaces Dataset"
mkdir -p sourcedata/opensurfaces
pushd sourcedata/opensurfaces
wget --progress=bar \
   http://labelmaterial.s3.amazonaws.com/release/opensurfaces-release-0.zip \
   -O opensurfaces-release-0.zip
unzip opensurfaces-release-0.zip
rm opensurfaces-release-0.zip
PROCESS=process_opensurfaces_release_0.py
wget --progress=bar \
  http://labelmaterial.s3.amazonaws.com/release/$PROCESS \
  -O $PROCESS
python $PROCESS
popd

fi


# ADE20K
if [ ! -f sourcedata/ade20k/ADE20K_2016_07_26/index_ade20k.mat ]
then

echo "Downloading ADE20K Dataset"
mkdir -p sourcedata/ade20k
pushd sourcedata/ade20k
wget --progress=bar \
   http://groups.csail.mit.edu/vision/sourcedatas/ADE20K/ADE20K_2016_07_26.zip \
   -O ADE20K_2016_07_26.zip
unzip ADE20K_2016_07_26.zip
rm ADE20K_2016_07_26.zip
popd

fi


# Now make broden in various sizes
if [ ! -f dataset/broden2_224/index.csv ]
then
echo "Building Broden2 224"
python src/joinseg.py --size=224
fi

if [ ! -f dataset/broden2_227/index.csv ]
then
echo "Building Broden2 227"
python src/joinseg.py --size=227
fi

if [ ! -f dataset/broden2_384/index.csv ]
then
echo "Building Broden2 384"
python src/joinseg.py --size=384
fi

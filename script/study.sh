echo python src/study.py \
  --layer dissection/caffe_reference_places365 conv1 \
  --layer dissection/caffe_reference_places365 conv2 \
  --layer dissection/caffe_reference_places365 conv3 \
  --layer dissection/caffe_reference_places365 conv4 \
  --layer dissection/caffe_reference_places365 conv5 \
  --categories object scene part material texture color \
  --top_n 5 \
  --imscale 144 \
  --barscale 1000 \
  --outdir layer_study

python src/study.py \
  --layer dissection/caffe_reference_places365 conv5 \
  --layername 'Alexnet' \
  --layer dissection/googlenet_places365 inception_5b/output \
  --layername 'Googlenet' \
  --layer dissection/vgg16_places365 conv5_3 \
  --layername 'VGG-16' \
  --layer dissection/resnet-152-torch-places365 caffe.Eltwise_510 \
  --layername 'Resnet' \
  --categories object scene part material texture color \
  --top_n 0 \
  --show_labels 0 \
  --imscale 100 \
  --barscale 1000 \
  --outdir arch_study

echo python src/study.py \
  --layer dissection/caffe_reference_places365 conv5 \
  --layer dissection/caffe_reference_imagenet conv5 \
  --layer dissection/weakly_videotracking conv5 \
  --layer dissection/weakly_audio conv5 \
  --layer dissection/weakly_solvingpuzzle conv5_s1 \
  --layer dissection/weakly_egomotion cls_conv5 \
  --categories object scene part material texture color \
  --top_n 5 \
  --imscale 144 \
  --barscale 1000 \
  --outdir super_study
#  --layer dissection/weakly_colorization conv5 \

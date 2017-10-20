#  --categories object scene part material texture color \
python src/study.py \
  --layername 'Resnet-152<br>(Places)' \
  --layer dissection/resnet-152-torch-places365 caffe.Eltwise_510 \
  --layername 'Densenet-161<br>(Places)' \
  --layer dissection/densenet161_places365 features \
  --layername 'Resnet-152<br>(Imagenet)' \
  --layer dissection/resnet-152-torch-imagenet caffe.Eltwise_510 \
  --layername 'Densenet-161<br>(Imagenet)' \
  --layer dissection/densenet161_imagenet features \
  --categories object \
  --top_n 0 \
  --show_labels 1 \
  --show_leftaxis 1 \
  --show_uniquecount 0 \
  --textsize 12 \
  --vmargin 12 \
  --imscale 80 \
  --barscale 800 \
  --outdir detail_study

#  --layername 'Alexnet<br>(Places)' \
#  --layer dissection/caffe_reference_places365 conv5 \
#  --layername 'Alexnet<br>(Imagenet)' \
#  --layer dissection/caffe_reference_imagenet conv5 \

python src/study.py \
  --layername 'Resnet-152<br>(Places)' \
  --layer dissection/resnet-152-torch-places365 caffe.Eltwise_510 \
  --layername 'Googlenet<br>(Places)' \
  --layer dissection/googlenet_places365 inception_5b/output \
  --layername 'VGG-16<br>(Places)' \
  --layer dissection/vgg16_places365 conv5_3 \
  --layername 'AlexNet-GAPWide<br>(Places)' \
  --layer dissection/alexnet_GAPallwide conv5 \
  --layername 'Alexnet<br>(Places)' \
  --layer dissection/caffe_reference_places365 conv5 \
  --layername 'Alexnet<br>(Imagenet)' \
  --layer dissection/caffe_reference_imagenet conv5 \
  --layername 'Alexnet<br>(Video Tracking)' \
  --layer dissection/weakly_videotracking conv5 \
  --layername 'Alexnet<br>(Ambient Sound)' \
  --layer dissection/weakly_audio conv5 \
  --layername 'Alexnet<br>(Puzzle Solving)' \
  --layer dissection/weakly_solvingpuzzle conv5_s1 \
  --layername 'Alexnet<br>(Egomotion)' \
  --layer dissection/weakly_egomotion cls_conv5 \
  --categories object scene part material texture color \
  --top_n 0 \
  --show_labels 1 \
  --show_leftaxis 1 \
  --show_uniquecount 1 \
  --textsize 12 \
  --vmargin 12 \
  --imscale 100 \
  --barscale 2400 \
  --outdir arch_study

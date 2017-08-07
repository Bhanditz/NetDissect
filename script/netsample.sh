python src/netsample.py \
    --directory sample/resnet-152-torch-places365 \
    --stride 3 \
    --blobs caffe.Eltwise_510 \
    --output_mat 0 \
    --weights zoo/resnet-152-torch-places365.caffemodel \
    --definition zoo/resnet-152-torch-places365.prototxt \
    --mean 109.5388 118.6897 124.6901 \
    --dataset dataset/broden2_224 \


python src/viewsample.py \
    --directory sample/resnet-152-torch-places365 \
    --blobs caffe.Eltwise_510 \
    --dataset dataset/broden2_224 \


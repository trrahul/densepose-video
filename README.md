# Densepose on video

Put the .py files with your detectron installation.

##### Use command 
```
  python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet50_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --wts https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x-e2e.pkl 
```
Note: Requires about 4GB VRAM
    
##### Tested with 

* Ubuntu 16.04
* Cuda 9.0
* CUDNN 6
* opencv 3.4

##### Example, ran on Geralt doing some witcherin'
![examples/geralt.gif](examples/geralt.gif)


##### And a bit of Night city.
![examples/cpunk.gif](examples/cpunk.gif)

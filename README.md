# EDMF
Event camera,Multimodal,Image Deblurring


### Real-time system
TODO


### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 

```python
python 3.11.5
pytorch 2.1.1
cuda 11.5
```



```
git clone https://github.com/ice-cream567/EDMF
cd EDMF
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### <span id="dataset_section"> Dataset </span> 
Use GoPro events to train the model. If you want to use your own event representation instead of DSE, download GoPro raw events and use EDMF/scripts/data_preparation/make_voxels_esim.py to produce your own event representation.

GoPro with DSE: [BaiduYunPan](https://pan.baidu.com/s/1gwJNSEj6uDfDNg4KgzY8DQ?pwd=XQHo )

REBlur with DSE: [BaiduYunPan](https://pan.baidu.com/s/1EMfa-nCrdUOpkIResTrQlw?pwd=A4x8 )

We also provide scripts to convert raw event files to SCER using scripts in [./scripts/data_preparation/](./scripts/data_preparation/). You can also design your own event representation by modify the script. Raw event files are provided by [EFNet](https://github.com/AHupuJR/EFNet) and can be downloaded:

GoPro with raw events: [[ETH_share_link](https://data.vision.ee.ethz.ch/csakarid/shared/EFNet/GOPRO_rawevents.zip)]  [[BaiduYunPan](link)/TODO]

REBlur with raw events: [[ETH_share_link](https://data.vision.ee.ethz.ch/csakarid/shared/EFNet/REBlur_rawevents.zip)]  [[BaiduYunPan](link)/TODO]



### Train
---
* train

  * ```python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/GoPro/EFNet.yml --launcher pytorch```

* eval
  * Download [pretrained model](https://pan.baidu.com/s/1PMfcEg6SkV5_ssq8ne13Og?pwd=mvhu) to ./experiments/pretrained_models/EDMF-GoPro.pth
  * ```python basicsr/test.py -opt options/test/GoPro/EDMF.yml  ```




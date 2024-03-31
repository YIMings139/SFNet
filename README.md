# SFNet: A Lightweight Slim Focus Net for Self-Supervised Monocular Depth Estimation

SFNet can estimate a depth map from a single image.

We ran our experiments with torch==1.8.1+cu102,  Python 3.6.13 and Ubuntu 18.04. 

## Data Preparation

Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.


## Qualitative results on the KITTI dataset

We show the superiority of our SFNet as follows:
![](https://github.com/YIMings139/SFNet/blob/main/img/Qualitative_result.png?raw=true)


The rows (from up to bottom) are RGB images, and the results by [Monodepth2](https://github.com/nianticlabs/monodepth2), [R-MSFM6](https://github.com/jsczzzk/R-MSFM),  [DiFFNet](https://github.com/brandleyzhou/DIFFNet) , [Lite-mono](https://github.com/noahzn/Lite-Mono) and SFNet.


## Quantitative results on the KITTI dataset
![image](https://github.com/YIMings139/SFNet/blob/main/img/Quantitative_result.png?raw=true)




## KITTI Evaluation
You can evaluates SFNet with:

```shell
python evaluate_depth.py --load_weights_folder model
```

## License & Acknowledgement
The codes are based on  [Lite-mono](https://github.com/noahzn/Lite-Mono), [Monodepth2](https://github.com/nianticlabs/monodepth2). Please also follow their licenses. Thanks for their great works.

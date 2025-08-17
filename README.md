<div align='center'>
<h1>A Flexible-Scale Point Cloud Upsampling Network Based on Local Geometry Kernel Interpolation for 3D Rock Surface Data</h1>
</div>

![example](./vis.png) 

##  Installation 
Step1. Install requirements:
```
python == 3.11.9
torch == 2.3.0
CUDA == 12.2
numpy == 1.26.4
open3d == 0.19.0
einops ==0.8.0
scikit-learn==1.5.0
tqdm==4.66.5
h5py==3.11.0
```
Step2. Compile the C++ extension modules:
```
cd models/Chamfer3D
python setup.py install
cd ../pointops
python setup.py install
```


##  Data preparation 

Datasets can be download from here:

| original PU-GAN | PU1K | pre-processed PU-GAN |
|:-------------:|:---------------:|:-------------:| 
|  [here](https://github.com/liruihui/PU-GAN) | [here](https://github.com/guochengqian/PU-GCN) | [Google Drive](https://drive.google.com/drive/folders/14Rd1jaRvGQHJAWM7q_FgJiL9U8_M30qf?usp=drive_link)  |


* We provide a pre-processed PU-GAN testing set with multiple resolutions of GT point clouds.
* If you want to generate testing point clouds from mesh files by youself, please refer to [here](https://github.com/yunhe20/Grad-PU).

After data preparation, the overall directory structure should be:
```
│data/
├──PU-GAN/
│   ├──train/
│   ├──test/
│   │   ├──pugan_4x
│   │   ├──pugan_16x
│   │   ├──arbitrary_scale
│   │   ├──.......
├──PU1K/
│   ├──train/
│   ├──test/
```

##   Training 


Training models on PU-GAN (or PU1K) dataset:
```
python train.py --dataset pugan
```
or
```
python train.py --dataset pu1k
```
Results will be saved under ./output


##  Testing & Evaluation

We provide a pre-trained weights on PU1K dataset in "pretrain" folder:

#### Testing example:
```
# 4X upsampling on PU1K dataset
python test.py --dataset pu1k --input_dir ./data/PU1K/test/input_2048/input_2048/ --gt_dir ./data/PU1K/test/input_2048/gt_8192/ --ckpt ./pretrain/pu1k_best.pth  --upsampling_rate 4 --save_dir ./result/pu1k_4x

# 16X upsampling on PU-GAN dataset
python test.py --dataset pugan --input_dir ./data/PU-GAN/test_pointcloud/input_2048_16X/input_2048 --gt_dir ./data/PU-GAN/test_pointcloud/input_2048_16X/gt_32768 --ckpt ./pretrain/pu1k_best.pth  --upsampling_rate 16 --save_dir ./result/pugan_16x

* You can use our code to get CD value. To calculate HD and P2F value, please refer to [here](https://github.com/guochengqian/PU-GCN). 

#### Surface reconstruction:
```
python surf_recon.py --file_path xxx.xyz --save_path xxx.obj

## Acknowledgements
This repo is heavily based on [KPConvX](https://github.com/apple/ml-kpconvx), [Dis-PU](https://github.com/liruihui/Dis-PU). Thanks for their great work!
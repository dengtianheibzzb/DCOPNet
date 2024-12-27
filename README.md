# DCOP-Net
Code for paper： DCOP-Net: A dual-filter cross attention and onion pooling network for
few-shot medical image segmentation

#### Abstract
Few-shot learning shows marvelous performance in medical image segmentation. 
However, existing few-shot medical image segmentation (FSMIS) models make it 
challenging to utilize the query image information fully, leading to prototype bias and 
low generalization ability of the models. To cope with these issues, we propose a dual-filter cross attention and onion pooling network (DCOP-Net) for FSMIS, which 
includes a prototype learning stage and a segmentation stage in its processing flow. 
Specifically, during the prototype learning stage, we design a dual-filter cross attention 
module (DFCA) to avoid the entanglement between query background (BG) features 
and support foreground (FG) features, thereby effectively integrating query FG features 
into support prototypes. We specially design an onion pooling module (OP) that 
combines eroding mask operations with masked average pooling (MAP) to learn 
multiple prototypes. This module retains the contextual information of support features 
within prototypes, alleviating the prototype bias issue. In the subsequent segmentation 
stage, we design a parallel threshold perceptual module (PTP) to enhance the quality of 
the threshold for distinguishing FG from BG. It is intended to process multiple inputs 
through multiple paths and reduce the impact of noise in the query image, thus ensuring 
accurate segmentation of the query image. Furthermore, we design a query self-reference regularization (QSR) that feeds query image information back into the query 
image itself to enhance the accuracy and consistency of the model in processing query 
images. Extensive experiments on three publicly available medical image datasets 
demonstrate that our method exhibits superior performance compared to state-of-the-art methods.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```
# Getting started

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

The pre-processed data and supervoxels can be downloaded by:
1) [Pre-processed CHAOS-T2 data and supervoxels](https://drive.google.com/drive/folders/1elxzn67Hhe0m1PvjjwLGls6QbkIQr1m1?usp=share_link)
2) [Pre-processed SABS data and supervoxels](https://drive.google.com/drive/folders/1pgm9sPE6ihqa2OuaiSz7X8QhXKkoybv5?usp=share_link)
3) [Pre-processed CMR data and supervoxels](https://drive.google.com/drive/folders/1aaU5KQiKOZelfVOpQxxfZNXKNkhrcvY2?usp=share_link)

### Training
1. Compile `./supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./supervoxels/setup.py build_ext --inplace`) and run `./supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/test.sh` 

### Acknowledgement
Code is based the works: [RPTNet](https://github.com/YazhouZhu19/RPT) ,[SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet) and [QNet](https://github.com/ZJLAB-AMMI/Q-Net)




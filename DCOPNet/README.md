# DCOP-Net
Code for paper： DCOP-Net: A dual-filter cross attention and onion pooling network for
few-shot medical image segmentation

#### Abstract
Few-shot learning has demonstrated remarkable performance in medical image segmentation. However, existing few-shot medical image segmentation (FSMIS) models often struggle to fully utilize query image information, leading to prototype bias and limited generalization ability. To address these issues, we propose the Dual-Filter Cross Attention and Onion Pooling Network (DCOP-Net) for FSMIS. DCOP-Net consists of a prototype learning stage and a segmentation stage. During the prototype learning stage, we introduce a Dual-Filter Cross Attention (DFCA) module to avoid entanglement between query background features and support foreground features, effectively integrating query foreground features into support prototypes. Additionally, we design an Onion Pooling (OP) module that combines eroding mask operations with masked average pooling to generate multiple prototypes, preserving contextual information and mitigating prototype bias. In the segmentation stage, we present a Parallel Threshold Perception (PTP) module to generate robust thresholds for foreground and background differentiation and a Query Self-Reference Regularization (QSR) strategy to enhance model accuracy and consistency. Extensive experiments on three publicly available medical image datasets demonstrate that DCOP-Net outperforms state-of-the-art methods, exhibiting superior segmentation and generalization capabilities.

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




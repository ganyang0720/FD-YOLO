
<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/7.png" width="400px">
</div>

# HM-YOLO: An Accurate Strawberry Ripeness Detection Model based on Hybrid Attention Mechanism and Multi-scale Progressive Feature Fusion

<div align="center">

  ![](https://img.shields.io/badge/python-3.10.13-red)
  [![](https://img.shields.io/badge/pytorch-2.0.1-red)](https://pytorch.org/)
  [![](https://img.shields.io/badge/torchvision-0.15.2-red)](https://pypi.org/project/torchvision/)
  
  
  

  [🛠️Installation Dependencies](https://blog.csdn.net/matt45m/article/details/134396179) |
  [🎤Introduction](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) |
 
  [👀Download Dataset](https://pan.baidu.com/s/16xYIAgHtBTwunHUBJjhT3Q?pwd=w85v)) |
  
  [🌊a Simple Anchor alignment metric](https://github.com/0811yu/0811yu.github.io) |
  [🚀Small Object detection improved](https://github.com/0811yu/0811yu.github.io) |
  
  [🤔Model generalization ability improved](https://github.com/0811yu/0811yu.github.io) |
 

</div>

## Dependencies:

 - Python 3.10.13
 - [PyTorch](https://pytorch.org/) 2.0.1
 - [Torchvision](https://pypi.org/project/torchvision/) 0.15.2
 - Ubuntu 22.04
 - NVIDIA GeForce RTX4090
   

## Introduction

This study presents an improved YOLOv8n model (HM-YOLO) designed to enhance the accuracy of machine recognition across various maturity stages of strawberries. Initially, a compact inverted module (PCIB) based on partial convolution (Pconv) was designed to replace the Bottleneck structure in the backbone network’s C2f module, thereby augmenting the expression capability of strawberry target features. Concurrently, an efficient partial hybrid attention mechanism (PHAM) was established by coupling lightweight multi-head self-attention (LMSA) with efficient local attention (ELA), which improved the model’s ability to capture long-range dependencies and accurately localize strawberry regions. Subsequently, within the neck network architecture, adaptively spatial feature fusion (ASFF) technology and an efficient DySample upsampler were introduced to construct a multi-scale progressive feature pyramid network (MS-FPN) integrated with PHAM, enabling precise identification and extraction of fine-grained features of strawberries of different sizes. Lastly, a Focaler-Shape-IoU loss function was proposed to address the variations in difficulty among strawberry samples and the impact of bounding box shapes and sizes on regression. A large complex dataset containing 16,800 strawberry images was built to validate the effectiveness of the model. Experimental results indicate that the proposed model achieves a detection precision of 92.1% and an mAP0.5 of 92.7%, reflecting improvements of 2.0% and 1.7% over the original model, respectively. Therefore, the improved model is deemed suitable to replace the baseline model for application in complex environments for detecting strawberries at different maturity stages.


</div>

## Overview of our network architecture
We propose a single-stage detection model (HM-YOLO) to improve the detection accuracy of strawberries at different maturities in complex environments. Firstly, the BottleNeck structure in the C2f module is replaced with PCIB in the backbone network to achieve efficient feature extraction. Secondly, MS-FPN is used instead of the original path aggregation network-feature pyramid networks (PAN-FPN) in the neck network to enable more efficient multi-scale feature fusion, which allows for precise identification and extraction of fine-grained features of strawberries at various maturities. Finally, the Focaler-Shape-IoU loss function in the prediction head part is adopted in place of the CIoU loss function to address the issues related to varying difficulty levels between strawberry samples and the impact of bounding box shape and size on regression. In the following sections, a detailed description of the work will be provided.
<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/4.png" width="700px">
</div>

## Ours strawberry datasets
 - StrawDI_Db1 Strawberry Dataset：This is a high-quality open-source dataset specifically designed for strawberries, sourced from approximately 150 hectares of strawberry plantations in Huelva Province, Spain, and includes 2,700 strawberry images.
 - Straw_Det Strawberry Dataset：A total of 1,500 strawberry images were collected from the Roboflow platform (https://robotflow.com/), primarily from the Strawberry-Detection Dataset and Strawberry-Detection2 Dataset.
 - Straw_Cul Strawberry Dataset：A total of 1,400 images depicting the growth process of strawberries were captured using a smartphone under various lighting conditions. The shooting scenes spanned both indoor and outdoor environments, with data collection occurring from 6:00 AM to 6:00 PM, encompassing various weather conditions such as sunny and overcast days.
 - SBDS Strawberry Dataset：This custom strawberry dataset, comprising a total of 5,600 images, was created by merging the self-captured Straw_Cul dataset with the open-source datasets Straw_Det and StrawDI_Db1.

## SBDS dataset example

<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/1.png" width="700px">
</div>

## Feature enhancement
To realistically simulate complex strawberry growth environments, various pixel-level and spatial transformations were employed for data augmentation, utilizing a range of image processing techniques from the Albumentations library (https://github.com/albumentations-team/albumentations). This approach significantly enhanced the expressiveness of the strawberry dataset. It is crucial to maintain the naturalness of image augmentation techniques to avoid deviating from actual growth patterns. For example, the rotation angle was restricted to between 5° and 15° to ensure that the rotation was not excessive while sufficiently simulating perspective variations in natural environments. Additionally, contrast adjustments were kept at a lower level, with a change rate not exceeding 0.15, to prevent over-sharpening or darkening of the images, thereby ensuring that the processed images better reflect natural conditions.The related techniques employed include:

1）RGB Shift: Randomly alters the order of the image color channels, enhancing the model's adaptability to color variations.

2）Shift Scale Rotate: Applies affine transformations to images, including translation, scaling, and rotation, enhancing the model's robustness to changes in target position and scale.
	
3）Hue Saturation Value: Randomly adjusts the hue and saturation of images, helping the model better adapt to different environments and scenes.

4）Random Brightness Contrast: Randomly adjusts the brightness and contrast of images to improve the model's adaptability to various lighting conditions.

5）Channel Shuffle: Randomly shuffles the RGB channels of images, adjusting the color distribution to help the model better recognize different color display methods.

6）Elastic Transform: Simulates the effect of images being distorted by elastic materials, aiding the model in learning to recognize non-rigid deformations that may be encountered in practical applications.


</div>

## Result of experiment
Results of ablation experiments on SBDS dataset, where ① represents PCIB, ② represents MS- FPN, ③ represents Focaler-Shape-IoU.
<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/8.png" alt="ddq_arch" width="700">
</div>

## Comparison of different datasets
The improved HM-YOLO model was evaluated using four different strawberry datasets to assess its performance in recognizing strawberries at various stages of ripeness, and its detection performance was compared with the YOLOv8n baseline model. The HM-YOLO model achieved improvements of 1.6% and 1.2% in the mAP0.5 and mAP0.5:0.95 metrics, respectively, on the SBDS dataset, which has the largest number of samples. Additionally, the precision and recall of the HM-YOLO model increased across all four datasets. Notably, the highest accuracy improvement of 2.5% was observed on the Straw_Cul dataset, while the most significant recall improvement of 1.5% was seen on the Straw_Det dataset. This phenomenon may be related to the greater impact of individual samples on overall performance in datasets with fewer samples. Overall, the HM-YOLO model significantly outperformed the baseline model in key performance metrics such as precision, recall, and mAP, demonstrating that the improved model is well-suited to replace the baseline model for detecting strawberries at different ripeness stages in relatively complex environments.
<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/5.png" alt="ddq_arch" width="700">
</div>

## Comparison with state-of-the-art models
To comprehensively evaluate the performance of the HM-YOLO model in strawberry ripeness detection, a comparison was conducted between the HM-YOLO algorithm and current leading object detection algorithms on the SBDS strawberry dataset. Additionally, a set of special experiments was conducted, using YOLOv8n as the baseline model and comparing it with state-of-the-art backbone networks such as MobileNetV4, EfficientViT, and GhostNetv2. Although Libra R-CNN excels in performance with mAP0.5 and mAP0.5:0.95 reaching 87.6% and 71.8% respectively, its parameter count is as high as 34.3M and its computational complexity reaches 293.6G, resulting in lower computational efficiency for this two-stage model. YOLOv10n, an end-to-end network, has a parameter count of only 2.3M and a computational complexity of 6.7G, but its performance on mAP0.5 and mAP0.5:0.95 is 0.1% and 0.3% lower than YOLOv8n, respectively. This may be due to YOLOv10n’s use of inexpensive depth-wise convolutions and point-wise convolutions to replace traditional convolutional downsampling and reduce the overhead of the classification head, which improves efficiency but sacrifices some performance. Furthermore, although our model shows only marginal performance improvements compared to other single-stage YOLO algorithms, such as a 2.6% higher mAP0.5 compared to the classic YOLOv5n, and a 2.0% higher mAP0.5 compared to the advanced YOLOv9-tiny, it exhibits a clear advantage when compared with models using lightweight backbone networks, particularly MobileNetV4, EfficientViT, and GhostNetv2. Notably, our model outperforms the best-performing YOLOv8-GhostNetv2 by 4.6% in the critical mAP0.5 metric. These experimental results demonstrate that despite having fewer parameters and lower FLOPs, our model still delivers superior detection performance. This advantage not only underscores the efficiency of our model but also highlights its feasibility in practical complex scenarios.
<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/6.png" alt="ddq_arch" width="1000">
</div>

**Note: Models marked with an asterisk (*) represents two-stage networks, those marked with a pound sign (#) denotes end-to-end networks,and those marked with an ampersand (&) are models equipped with other backbone networks, and unmarked models are classified as single-stage network.**

To more intuitively demonstrate the performance of our model in detecting strawberries at different stages of ripeness, three representative networks were selected for comparison on the SBDS dataset: the two-stage network Libra R-CNN, the advanced end-to-end YOLOv10n from the YOLO series, and the single-stage network equipped with the outstanding backbone network GhostNetv2. Subfigures (a), (b), (c), and (d) represent YOLOv8n-GhostNetv2, YOLOv10n, Libra R-CNN, and our model, respectively. YOLOv8n-GhostNetv2 failed to identify small strawberries marked as positions r and s in the detection images, indicating that models using GhostNetv2 as a backbone network are not suitable for detecting strawberries in the early, unripe stage. YOLOv10n, on the other hand, incorrectly recognized two overlapping strawberries at position t as a single strawberry and missed the strawberry marked as position u, demonstrating poor performance in strawberry detection in complex scenes. The two-stage network Libra R-CNN failed to detect strawberries marked as positions v and w, and erroneously identified a semi-ripe strawberry marked as position x as an unripe strawberry, highlighting limitations in recognizing the turning stage strawberries. In contrast to these existing models, our model successfully and accurately identified strawberries at three different stages of ripeness in complex scenarios, demonstrating substantial application potential.

<div align="center">
<img src="https://github.com/ganyang0720/HM-YOLO/blob/main/Main%20image/3.png" alt="ddq_arch" width="700">
</div>

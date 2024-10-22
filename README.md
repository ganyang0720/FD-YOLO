
<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/ganyang0720-patch-1/Fig.6.jpg" width="400px">
</div>

# FD-YOLO: A Real-Time End-to-End Face Mask Detection Algorithm Based on Attention Mechanism

<div align="center">

  ![](https://img.shields.io/badge/python-3.10.13-red)
  [![](https://img.shields.io/badge/pytorch-2.2.1-red)](https://pytorch.org/)
  [![](https://img.shields.io/badge/torchvision-0.17.2-red)](https://pypi.org/project/torchvision/)
  
  
  

  [🛠️Installation Dependencies](https://blog.csdn.net/matt45m/article/details/134396179) |
  [🎤Introduction](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) |
 
  [👀Download Dataset](https://pan.baidu.com/s/1EfLKnzaOP_Bjy1H8oB8k0g?pwd=8b7e)) |
  
  [🌊a Simple Anchor alignment metric](https://github.com/0811yu/0811yu.github.io) |
  [🚀Small Object detection improved](https://github.com/0811yu/0811yu.github.io) |
  
  [🤔Model generalization ability improved](https://github.com/0811yu/0811yu.github.io) |
 

</div>

## Dependencies:

 - Python 3.10.13
 - [PyTorch](https://pytorch.org/) 2.2.1
 - [Torchvision](https://pypi.org/project/torchvision/) 0.17.2
 - Ubuntu 20.04
 - NVIDIA GeForce RTX4090
   

## Introduction

Unmanned ground vehicle detection systems are capable of efficiently implementing real-time detection of face masks. However, the existing detection algorithms are limited in their deployment in practical applications due to their high computational costs and difficulty in ensuring accuracy. To address this, we developed a lightweight end-to-end real-time face mask detection algorithm based on YOLOv10n. Firstly, a compact and efficient module(Pstar block) based on the "star operation" (element-wise multiplication) was designed to replace the Bottleneck structure in the C2f module of the backbone network, thereby significantly enhancing the feature extraction capability of the model. Secondly, a lightweight collaborative attention module (LCAM) was proposed, allowing the model to better focus on the position of face mask targets and their relationships in complex scenes, thereby improving its detection capability in dense crowds. Additionally, a lightweight localization feature pyramid network (L2-FPN) based on an efficient local attention (ELA) module was developed. The ELA module filters out low-level feature information and enhances the localization ability for face masks. Then, the features screened for their positional information are fused with multi-scale features, remarkably enhancing the representation of face mask targets of different scales.Finally, the Focaler-UIoU loss function was proposed, effectively addressing the issue of imbalanced sample difficulty in face mask detection and improving the detection performance of the model in densely occluded scenes. Experimental results show that our model achieves a precision of 75.7% and an mAP50 of 74.1%, respectively, while the model parameters and FLOPs are reduced by 0.2M and 0.3G, respectively. To verify the generalization capability of the model., tests were conducted on the AIZOO and RFMD public datasets, and the results show that the model improves the mAP50 metric by 2.6% and 2.2% over the baseline model, respectively. Moreover, when deployed on the embedded edge computing device Jetson AGX Orin, the detection speed reaches 79.1FPS under FP16 quantization, further verifying the superior performance of our model. In summary, the proposed model not only enables real-time and accurate recognition of face mask targets but also effectively balances accuracy and speed.


</div>

## Overview of our network architecture
An end-to-end lightweight face mask detection model (FD-YOLO) was proposed, and its structure is illustrated in Fig. 1. First, in the backbone network, the BottleNeck structure in the C2f module was replaced with the Pstar Block to enable more efficient feature extraction. Second, in the neck network, the computationally expensive PAN-FPN structure was replaced with the L2-FPN module to achieve more efficient multi-scale feature fusion, allowing for precise identification and localization of face mask target features. Finally, in the prediction head, the less efficient CIoU loss function was replaced with the Focaler-UIoU loss function, which effectively addressed the differences in difficulty among face mask samples and the impact of predicted bounding box quality on regression. The details of this research work will be presented next.
<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/photos/Fig.1.png" width="700px">
</div>

## Ours datasets
 To verify the robustness of the proposed method, the model was evaluated on four different large datasets. The first dataset is the AIZOO open-source face mask dataset (https://github.com/AIZOOTech/FaceMaskDetection), which consists of two sub-datasets, WIDER Face and MAFA, with a total of 7,872 images, including 6,057 in the training set and 1,815 in the validation set. The second dataset is an internet dataset, where 1,200 diverse face mask photos were collected from the web, including 800 images in the training set and 400 images in the test set. The third dataset is the RFDM open-source dataset (https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset), from which 5,000 photos were selected and randomly divided into a training set of 3,400 images and a validation set of 1,600 images. The fourth dataset is the self-made Largest Face Mask Dataset (LFMD). This custom dataset was created by combining face mask images taken on campus with images from the internet and the RFDM open-source dataset to expand the data volume. Ultimately, this dataset contains 10,200 photos, including 15,242 mask images and 6,834 face images. There are approximately 4,000 face mask photos taken on campus, all captured by smart devices. The randomly divided training and validation sets contain 8,160 and 2,040 images, respectively. Please refer to the links in the configuration section to access the face mask datasets. 

## LFMD dataset example

<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/photos/Fig.4.png" width="700px">
</div>



## Result of experiment
Results of ablation experiments on LFMD dataset, where ① represents Pstar, ② represents L2- FPN, ③ represents Focaler-UIoU.
<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/ganyang0720-patch-1/Fig.8.png" alt="ddq_arch" width="700">
</div>

## Comparison of different datasets
The improved FD-YOLO model was evaluated on four different datasets for its performance in recognizing face mask targets, and its detection capabilities were compared against the baseline model YOLOv10n. On the LFMD dataset, which is the most extensive and complex, the enhancements in the FD-YOLO model were particularly significant, with improvements of 2.4% and 1.4% in the mAP50 and mAP95 metrics, respectively. This indicates that the improved model has stronger adaptability and generalization capabilities for different sample characteristics. Additionally, FD-YOLO showed improvements in both precision and recall across all four datasets. Specifically, on the AIZOO dataset, the precision of the model was increased by 2.5%; while on the Internet dataset, the recall rate saw the most significant improvement, reaching 1.5%. This phenomenon might be due to the fact that in datasets with fewer samples, changes in individual samples have a more noticeable impact on overall recall rates. Overall, despite the FPS being slightly reduced due to the introduction of multiple ELA modules, the improved model significantly outperformed the baseline model in key performance indicators such as precision, recall, and mAP. Notably, the lowest FPS achieved by the improved model on the Internet dataset was still 69.1, meeting real-time requirements. In summary, the improved FD-YOLO model can effectively replace the baseline model and is highly suitable for real-time detection of face mask targets in complex scenarios.
<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/ganyang0720-patch-1/Fig.9.png" alt="ddq_arch" width="700">
</div>

## Comparison with state-of-the-art models
To comprehensively evaluate the performance of our model in face mask detection, the FD-YOLO algorithm is compared with current leading object detection algorithms on the LFMD mask dataset. Furthermore, a series of special experiments are conducted using YOLOv10n as the baseline model, equipped with excellent lightweight backbone networks such as MobileNetV4, EfficientViT, GhostNetv2, and StarNet. Libra R-CNN exhibits outstanding performance in mAP50 and mAP95, achieving 71.1% and 31.3% respectively. However, its parameter count is as high as 34.3M, and its computational complexity reaches 293.6G, making this two-stage model computationally inefficient, difficult to deploy on resource-constrained edge devices. Similarly, the CenterNet model directly detects the center and size of objects without using anchors. Although its mAP50 reaches 59.0%, the large parameter scale and high computational complexity also make it unsuitable for real-time face mask detection. Although our model shows only slight advantages in performance improvements compared to other single-stage YOLO series algorithms, for example, it is 2.9% higher than the classic YOLOv5n in mAP50 and 2.1% higher than the advanced YOLOv11n, when compared with models using lightweight backbone networks, especially MobileNetV4, EfficientViT, GhostNetv2, and StarNet, our model demonstrates significant advantages. Particularly, in the critical indicator mAP50, our model outperforms the best-performing YOLOv8-MobileNetV4 by 3.1%. These experimental results indicate that our model not only has lower parameters and FLOPs but also exhibits superior detection performance. In summary, FD-YOLO achieves an ideal balance between accuracy and computational complexity, making it more suitable for real-time face mask object detection.
<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/ganyang0720-patch-1/Fig.7.png" alt="ddq_arch" width="1000">
</div>

**Note: Models marked with an asterisk (*) represents two-stage networks, those marked with a pound sign (#) denotes end-to-end networks,and those marked with an ampersand (&) are models equipped with other backbone networks, and unmarked models are classified as single-stage network.**

To visually demonstrate the outstanding performance of our model in detecting face mask targets, three representative networks were specifically selected for comparative testing on the LFMD dataset. These three networks are the currently popular single-stage network YOLOv8n, the cutting-edge model YOLO11n from the YOLO series, and the end-to-end network YOLOv10n-MobileNetv4, which is based on YOLO10n and integrates the excellent backbone network MobileNetv4. Detailed test results are presented in Fig. 11, where subfigures (a), (b), (c), and (d) correspond to YOLOv8n, YOLOv10n-MobileNetv4, YOLO11n, and our model, respectively.From the comparison, it was found that YOLOv10n-MobileNetv4 incorrectly detects faces as masks at “r” and misidentifies “s”, highlighting the insufficient feature extraction capability of models utilizing MobileNetv4 as the backbone. Meanwhile, although YOLOv8n successfully detects the face at “t”, it also misdetects “t” and “u,” indicating that its performance in detecting face mask targets in complex scenes still requires improvement. On the other hand, the YOLO11n model also incorrectly detects “v” and “w”. In contrast, our model, which is specifically designed for face mask detection, accurately identifies face mask targets in complex and dense scenes, further highlighting the unique advantages of our model in addressing challenges related to dense occlusions.

<div align="center">
<img src="https://github.com/ganyang0720/FD-YOLO/blob/ganyang0720-patch-1/Fig.10.png" alt="ddq_arch" width="700">
</div>

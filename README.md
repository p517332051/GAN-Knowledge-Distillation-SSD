### GAN-Knowledge Distillation for One-Stage Object Detection

​                                                                        HongWei

​                                                               517332051@qq.com

​                                                                    StartDt AI Lab  

#### **Method**

我们采用一步目标检测算法SSD作为我们的目标检测算法,SSD目标检测算法结构主要分成两部分，1）骨架网络,作为特征提取器。2）Head,在骨架网络提取的特征上，检测出目标的类别和位置。为了能获取更好的知识蒸馏效果，合理利用这个两个部分至关重要。

#####  Overall Structure

fig 1为我们算法模型的整体结构，我们首先使用一个容量更大的SSD模型，在充分训练后将该SSD模型拆分成骨架网络和SSD-Head，其中骨架网络作为teacher net，然后再挑选一个容量较小的CNN作为student net。我们把teacher net生成的多个feature map作为true sample，而student net生成的多个feature map作为fake sample，并且将true sample和fake sample送入D Net中相对应的每个判别网络(fig 2)中，同时把fake sample输入到SSD-Head中。![v2-02d1a039110fa229e6e6ed6a972731e3_r](v2-02d1a039110fa229e6e6ed6a972731e3_r.jpg)

**Figure 1:**Teacher Net和 SSD-Head分别为容量更大并且充分训练后的SSD模型的骨架网络和头部网络。Student Net为一个容量较小的网络。D Net为多个6个小型判别网络组成的模块。

![v2-099397f6658320288282fd9b65b235d2_r](v2-099397f6658320288282fd9b65b235d2_r.jpg)

**Figure 2:**为D-Net模块中其中一个判别网络，由多个下采样卷积层组成。

##### Training Process

如下面的公式。

![seq1](seq1.jpg)

![v2-099397f6658320288282fd9b65b235d2_r](seq2.png)

##### Results

我们的脚本工具为gluoncv，和mxnet。训练机器目前已经有4个1080ti。将原生的SSD和在不同的Teacher net下知识蒸馏的SSD做比较，最高可以提升student net 2.8mAP。不过有趣的是，当teacher net为ResNet101，student net为ResNet18时，提升的效果反而不如ResNet50。后面发现在gluoncv上ssd_ResNet101,发现其效果在voc上的mAP只有79+，而ssd_ResNet50mAP达到80+。由于COCO数据集实在庞大，由于硬件资源有限训练一个模型需要一周时间，所以只用ResNet50作为Teacher Net(如果有的小伙伴有更多的硬件资源，可以尝试使用SSD+FPN作为Teacher Net，也可以将本方法迁移到Faster RCNN等一切端到端的深度学习方法上进行测试，相信会有意想不到的结果。～_～训练了好几个月才这些结果)。

| Student net | Teacher net                                                  | Pascal Voc 2007 test                         |
| :---------- | :----------------------------------------------------------- | -------------------------------------------- |
| MobilenetV1 | -<br />VGG16<br />ResNet50<br />ResNet101<br />ResNet50(with cosine attention) | 75.4<br />77.3<br />77.7<br />77.6<br />78.4 |
| MobilenetV2 | -<br />VGG16<br />ResNet50<br />ResNet101<br />              | 75.9<br />77.2<br />77.7<br />77.5<br />     |
| ResNet18    | -<br />VGG16<br />ResNet50<br />ResNet101<br />              | 74.8<br />77.2<br />77.6<br />77.3<br />     |

Table 1. 不同的student net在不使用对抗知识蒸馏和使用对抗知识蒸馏在不同大小的teacher net的测试结果。训练sets为Pascal Voc 2007 and Pascal Voc 2012 trainval，测试sets为Pascal Voc 2007 test。



| Student net      | Teacher net           | COCO Val 2017        |
| :--------------- | :-------------------- | -------------------- |
| MobilenetV1_0.75 | -<br />ResNet50<br /> | 18<br />23<br />     |
| MobilenetV1      | -<br />ResNet50<br /> | 21.7<br />25.4<br /> |
| MobilenetV2      | -<br />ResNet50<br /> | 22<br />24.6<br />   |

Table 2. 使用COCO train 2017作为训练集，COCO val 2017作为测试集。



以上就是本方法的的结果报告，现在正在加入cosine attention机制目前在voc上所有的模型都可以提提升接近1个mAP，目前正在coco上实验。

目前已经将该方法使用在faster rcnn上,效果如下：

| Student net                                      | Teacher net            | Pascal Voc 2007 test       |
| :----------------------------------------------- | :--------------------- | -------------------------- |
| ResNet50+roipooling（with GAN-KD）               | -<br />ResNet101<br /> | 67.0<br />72.0(+5)<br />   |
| ResNet50+roialign（with GAN-KD）                 | -<br />ResNet101<br /> | 71.7<br />74.0(+2.3)<br /> |
| ResNet50（https://arxiv.org/pdf/1906.03609.pdf） | -<br />ResNet101<br /> | 69.0<br />72.0(+3)<br />   |

**Table 3.**  Teacher net为骨架网络为ResNet101的faster rcnn，且使用Pascal Voc 2007 trainval作为训练集，在Pascal Voc 2007 test测试集上mAP为74.8+。第一行和第二行使用GAN Knowledge Distillation方法，第三行为cvpr2019的 Distilling Object Detectors with Fine-grained Feature Imitation的方法效果。




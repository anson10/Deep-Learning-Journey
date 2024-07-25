# Convolutional Neural Network (CNN) Architectures

Convolutional Neural Networks (CNNs) have been instrumental in advancing the field of computer vision. Various architectures have been developed to improve the performance and efficiency of CNNs. Below is a summary of some of the most influential CNN architectures:

## 1. LeNet-5 (1998)
- **Developed by:** Yann LeCun et al.
- **Application:** Handwritten digit recognition (MNIST dataset).
- **Architecture:**
  - Input: 32x32 grayscale image.
  - Layers: Convolutional -> Subsampling -> Convolutional -> Subsampling -> Fully Connected -> Fully Connected -> Output.
- **Key Features:** Introduced the concept of convolutional and pooling layers.
- ![LeNet-5 Architecture](https://www.datasciencecentral.com/wp-content/uploads/2021/10/1lvvWF48t7cyRWqct13eU0w.jpeg)

## 2. AlexNet (2012)
- **Developed by:** Alex Krizhevsky et al.
- **Application:** Image classification (ImageNet dataset).
- **Architecture:**
  - Input: 227x227 RGB image.
  - Layers: 5 Convolutional + Max-Pooling layers -> 3 Fully Connected layers.
- **Key Features:** Utilized ReLU activation, dropout for regularization, and GPU acceleration.
- ![AlexNet Architecture](https://miro.medium.com/v2/resize:fit:1400/1*0dsWFuc0pDmcAmHJUh7wqg.png)

## 3. VGGNet (2014)
- **Developed by:** Karen Simonyan and Andrew Zisserman.
- **Application:** Image classification and localization.
- **Architecture:**
  - Input: 224x224 RGB image.
  - Layers: 16-19 layers deep with small 3x3 filters, followed by Fully Connected layers.
- **Key Features:** Demonstrated the importance of depth in CNNs.
- ![VGGNet Architecture](https://www.researchgate.net/profile/Timea-Bezdan/publication/333242381/figure/fig2/AS:760979981860866@1558443174380/VGGNet-architecture-19.ppm)

## 4. GoogLeNet (Inception v1) (2014)
- **Developed by:** Christian Szegedy et al.
- **Application:** Image classification (ImageNet dataset).
- **Architecture:**
  - Input: 224x224 RGB image.
  - Layers: 22 layers deep with Inception modules.
- **Key Features:** Introduced Inception modules to capture multi-scale features.
- ![GoogLeNet Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20200429201304/Incepption-module.PNG)

## 5. ResNet (2015)
- **Developed by:** Kaiming He et al.
- **Application:** Image classification, detection, and localization.
- **Architecture:**
  - Input: 224x224 RGB image.
  - Layers: 34, 50, 101, 152 layers deep with residual blocks.
- **Key Features:** Introduced residual connections to tackle the vanishing gradient problem.
- ![ResNet Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20200424011138/ResNet.PNG)

## 7. Xception (2016)
- **Developed by:** François Chollet.
- **Application:** Image classification.
- **Architecture:**
  - Input: 299x299 RGB image.
  - Layers: Depthwise separable convolutions with residual connections.
- **Key Features:** Extreme version of Inception, using depthwise separable convolutions.
- ![Xception Architecture](https://www.researchgate.net/profile/Rodrigo-Salas-5/publication/357973065/figure/fig2/AS:1127438167879680@1645813614280/Depthwise-separable-convolution-of-the-MobileNet-which-factorizes-the-convolution-into.ppm)

## 8. SENet (2018)
- **Developed by:** Jie Hu et al.
- **Application:** Image classification.
- **Architecture:**
  - Input: Variable image sizes.
  - Layers: Squeeze-and-Excitation blocks integrated into existing architectures.
- **Key Features:** Introduced Squeeze-and-Excitation (SE) blocks to recalibrate channel-wise feature responses.
- ![SENet Architecture](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-14_at_9.11.02_PM.png)

# OTHER ARCHITECTURES:

## 1. MobileNet (2017)
- **Developed by:** Andrew G. Howard et al.
- **Application:** Mobile and embedded vision applications.
- **Architecture:**
  - Input: 224x224 RGB image.
  - Layers: Depthwise separable convolutions.
- **Key Features:** Optimized for mobile devices with reduced computation and memory usage.

## 2. EfficientNet (2019)
- **Developed by:** Mingxing Tan and Quoc V. Le.
- **Application:** Image classification and transfer learning.
- **Architecture:**
  - Input: Variable image sizes.
  - Layers: Scaling width, depth, and resolution systematically.
- **Key Features:** Achieved state-of-the-art performance with efficient scaling.

## 3. DenseNet (2016)
- **Developed by:** Gao Huang et al.
- **Application:** Image classification.
- **Architecture:**
  - Input: 224x224 RGB image.
  - Layers: Dense blocks where each layer is connected to every other layer.
- **Key Features:** Improved parameter efficiency and feature reuse.

## Summary
These CNN architectures have significantly contributed to the advancement of computer vision tasks. Each architecture introduced novel concepts and improvements, paving the way for more efficient and accurate models in various applications.

### References
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
4. Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going deeper with convolutions.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
6. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks.
7. Howard, A. G., Zhu, M., Chen, B., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications.
8. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
9. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions.
10. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.

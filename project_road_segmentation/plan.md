# Plan for the Project


## 0. ressources
example of API doing what we want : https://developers.arcgis.com/python/sample-notebooks/automatic-road-extraction-using-deep-learning/
Papers doing what we are interested in [[Road segmentation Using CNNwith GRU]](https://arxiv.org/pdf/1804.05164.pdf)

Previous projects : [[U-Net]](https://blog.alexandrecarlier.com/projects/unet/)

**Papers**
- [[U-Net: Convolutional Networks for Biomedical Image Segmentation]](https://arxiv.org/pdf/1505.04597v1.pdf)
- [[DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation]](https://ieeexplore.ieee.org/abstract/document/9183321)
- [[Comparison of Different Convolutional Neural Network Architectures for Satellite Image Segmentation]](https://ieeexplore.ieee.org/abstract/document/8588071)



## 1) Image treatment for better analysis

Set all images to grayscale to reduce memory usage
Maybe reduce the pixelation to reduce memory usage (tradeoff, we may have a lower F-1 score). The pixels'values are equal to the greater pixels' values (transfomr back)



## 2) Data augmentation
Use api [[scikit-image (skimage)]](https://scikit-image.org/docs/stable/api/skimage.html).

Possible combinations : https://www.codespeedy.com/image-augmentation-using-skimage-in-python/. 
* Rotation of the pictures 
* sheared images
* wraping
* blurring
* brightness
* contrast
* image flipping
* generated corruptions

K-fold

## 3) CNN

choice to make on the activation function



To increase performance, Implement : 
* node Drop Out
* batch normalisation
* 


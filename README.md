# Project
An end-to-end machine learning pipeline built with python and tensorflow (keras). The goal is to categorise the presence (and location) of a dog within a room from an IoT camera. A custom image classifier is needed in this instance to learn the specific locations within the room. Once trained, the model is served on Google Cloud (Cloud Run) such that it can be called from a Home Assistant (HA) server, with the information ultimately visible on a dashboard, accessible by tablet, phone, or even smartwatch.

## Details - Latest
The pipeline is specifically an example for image classification, with a custom Residual Neural Network **(ResNet)**. The custom ResNet is 12 layers deep, a cut-down custom architecture modelled off of the popular ResNet-50. After experimentation, ResNet models seemed to significantly outperform even deep CNNs for the same training set. The latest model was trained on ~1000 raw images of dogs within a room, ~5000 after image augmentation is applied. It achieves **94%** accuracy on a roughly balanced test dataset, for a model outputting **4** classes.

A new addition is the ability to quickly relabel images on-the-fly (human-in-the-loop). This can be triggered from smartwatch or phone via HA, with images stored in labelled folders in the cloud.

Thanks and credit to [priya-dwivedi](https://github.com/priya-dwivedi/Deep-Learning/commits?author=priya-dwivedi) for the [tutorial on building blocks for the ResNet](https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb).

## Details - May 2023
The pipeline is specifically an example for image classification, with a custom Convolutional Neural Network **(CNN)**. The latest model was trained on ~1000 raw images of dogs within a room, ~3000 after image augmentation is applied. It achieves **79%** accuracy on a balanced test dataset, for a model outputting **4** classes.

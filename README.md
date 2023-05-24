# Project
An end-to-end machine learning pipeline built with python and tensorflow (keras). The goal is to categorise the presence (and location) of a dog within a room from an IoT camera. A custom image classifier is needed in this instance to learn the specific locations within the room. Once trained, the model is served on Google Cloud (Cloud Run) such that it can be called from a Home Assistant server, with the information ultimately visible on a dashboard, accessible by tablet, phone, or even smartwatch.

## Details
The pipeline is specifically an example for image classification, with a custom Convolutional Neural Network **(CNN)**. The latest model was trained on ~1000 raw images of dogs within a room, ~3000 after image augmentation is applied. It achieves **79%** accuracy on a balanced test dataset, for a model outputting **4** classes.

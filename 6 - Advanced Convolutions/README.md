**Session-6: Advanced Convolutions** 

Assignment: 

Fix the network: https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw

   ![assignment](https://user-images.githubusercontent.com/60026221/217057292-3ec2cb8a-79f4-4494-ab39-33c8d1116037.JPG)

* As of need, 4 convolution blocks are used with total of 12 layers in-it. The architecture also contains the dilation convolution (1-layer) and depthwise (1-layer) along with pointwise(1-layer). It summarizes as 9 convolution layers + 1 Dilation layer + 1 Depthwise seperable convolutions are used and they made up the number of **parameters as: 1,91,514**

* The repo contains 
   A. main.py where the necessary training & testing happens and returns scores.
   B. model.py which has the architecture.
   C. utils.py which has useful functions like normalizing, dataloading, briefly called as helper functions and also returns class accuracy
   D. 2 Notebooks one with more 200K parameters which provided good result but have overfitting, which is trained without using any special features and other than less than 200K parameters which used special features.
   

1. As a First step, we have to know about data (CIFAR-10):

2. Images:

  * Below, you can see few of training images which are augmented with cutout/shiftscaleRotate/Horizontal Flip
  
    ![train](https://user-images.githubusercontent.com/60026221/217634262-a1666ba4-d650-4b5b-a5cb-71adb7aaea40.png)

  * Below, you can see few of testing images which should be augmented as we use this for validations/testing.
  
    ![test](https://user-images.githubusercontent.com/60026221/217634284-1ce4c361-8fc9-45b8-8930-2fb6dc2ffad2.png)










# Proceedings to add in README.md: Need to add plots & images with augmentations & explain albumentations and briefly about sessions.

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
      
     * Documentation:  https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
     * The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
     * Baseline Model from PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    

2. Images:

     * Below, you can see few of training images which are augmented with cutout/shiftscaleRotate/Horizontal Flip

       ![train](https://user-images.githubusercontent.com/60026221/217634262-a1666ba4-d650-4b5b-a5cb-71adb7aaea40.png)

     * Below, you can see few of testing images which should be augmented as we use this for validations/testing.

       ![test](https://user-images.githubusercontent.com/60026221/217634284-1ce4c361-8fc9-45b8-8930-2fb6dc2ffad2.png)

3. Architecture is on GPU and is of format: 
   
    * 4 Blocks of Convs + GAP LAYER
       
       * Block1: Conv1(3X3X16) with padding + Conv2(3x3x32) + Conv3(3x3x32) 
       * Block2: Conv4(5x5x128 i,e,,kernel incl. of dilation) with dilated convolution (Dilation=2) + Conv5(3x3x64) + Conv6(3x3x32) with stride of 2 and padding
       * Block3: [Conv7(3x3x128)+Conv8(1x1x256] a depthwise convolution + Conv9(3x3x32) 
       * BLock4: Conv10(3x3x32) + conv11(1x1x16) + conv12(3x3x10) 
       * GAP LAYER : Kernel_size = 3
    * There are total of 187,418 Parameters ~ 187K 
    
4. Ran For 84 Epochs in CIFAR10_model.ipynb.
5. Analysis: 
    * With out augmentation reached training accuracy of 86% and val-accuracy of 77%, as it seems to be ok but not good. There is a need to avoid overfitting and add variance to data. Hence, we have procedded with augmenting the images using albumentations library. 
    * With augmentation, it performed quite well. 
7. Got the accuracies of:










# Proceedings to add in README.md: Need to add plots & images with augmentations & explain albumentations and briefly about sessions.

  # CIFAR-10 (CIFAR10.ipynb)
                                                        
                                                        
                                                        
                                                  
  **Session-6: Advanced Convolutions** : Describing about normal convolutions and its expensive computations, introduction of dilation convolution and depthwise convolutions and group convolutions and their expensive computation compared to other type of convolution and their usage and their methodologies.
                         
                                                      

Assignment: 

Fix the network: https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw

   <img src="https://user-images.githubusercontent.com/60026221/217057292-3ec2cb8a-79f4-4494-ab39-33c8d1116037.JPG" width=40% height=40%>


### The repo contains 
      
      A. main.py where the necessary training & testing happens and returns scores
      
      B. model.py which has the architecture built using PyTorch

      C. utils.py which has useful functions like normalizing, dataloading, briefly called as helper functions and also returns class accuracy

      D. CIFAR10.ipynb is the one which has given the target val-accuracy with receptive Field greater than 44.
      
    
      
Synopsis:
-----------------

1. As a First step, we have to know about data (CIFAR-10):
      
     * Documentation:  https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
     * The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
     * Baseline Model from PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    

2. Images:

     * Below, you can see few of training images which are augmented with cutout/shiftscaleRotate/Horizontal Flip

       <img src="https://user-images.githubusercontent.com/60026221/217634262-a1666ba4-d650-4b5b-a5cb-71adb7aaea40.png" width = 140% height = 140%>

     * Below, you can see few of testing images which should be augmented as we use this for validations/testing.

       <img src="https://user-images.githubusercontent.com/60026221/217634284-1ce4c361-8fc9-45b8-8930-2fb6dc2ffad2.png" width=140% height = 140%>

3. Architecture is on GPU and is of format: 
   
    * 4 Blocks of Convs + GAP LAYER
       
       * Block1: [Conv1(3X3X16) with padding + Conv2(3x3x32) with Padding + Conv3(3x3x32) with Padding & stride]
       * Block2: [Conv4(5x5x128 i,e,kernel incl. of dilation2) with dilated convolution (Dilation=2) + Conv5(3x3x64) + Conv6(3x3x32) with stride of 2 and padding]
       * Block3: [[Conv7(3x3x128)+Conv8(1x1x256] a depthwise convolution ]
       * BLock4: [Conv10(3x3x32) + conv11(1x1x16) + conv12(1x1x10)]
       * GAP LAYER : Kernel_size = 1
       
    * There are total of **170106 Parameters ~ 170K**
    
    * Has Receptive Field of 47.
    
4. Ran For 84 Epochs in CIFAR10.ipynb.

6. Analysis: 

    * Base Model: **less_weighted-model.ipynb:** With out augmentation reached training accuracy of 86% and val-accuracy of 77%, as it seems to be ok but not good. There is a need to avoid overfitting and add variance to data. Hence, we have procedded with augmenting the images using albumentations library and this notebook architecture is  4 convolution blocks are used with total of 12 layers in-it. The architecture also contains the dilation convolution (1-layer) and depthwise (1-layer) along with pointwise(1-layer). It summarizes as 9 convolution layers + 1 Dilation layer + 1 Depthwise seperable convolutions are used and they made up the number of **parameters as: 1,91,514**
     
    
    * Improved model: **CIFAR10.ipynb** With augmentation, it performed quite well but has to be improved more.
    
    
7. Accuracies: 
   
    * Model:
    ---------
      **CIFAR10.ipynb**
                 
          EPOCH: 84/84, 85 is achieved at Epoch 59. The last two epoch accuracies are shown below
        
                EPOCH: 82
                Loss=0.6692079305648804 Batch_id=390 train-Accuracy=80.47: 100%|██████████| 391/391 [00:12<00:00, 32.05it/s] 

                Test set: Average loss: 0.0032, val-Accuracy: 8615/10000 (86.15%)

                EPOCH: 83
                Loss=0.465636670589447 Batch_id=390 train-Accuracy=81.17: 100%|██████████| 391/391 [00:12<00:00, 32.02it/s]  

                Test set: Average loss: 0.0033, val-Accuracy: 8570/10000 (85.70%)
      
 
              
 8. Plots: 
 
      ![accuracy](https://user-images.githubusercontent.com/60026221/217802973-2802314e-6c37-4af6-a159-6323b4163d38.JPG)
 
 
 
      ![loss](https://user-images.githubusercontent.com/60026221/217802984-4b8ec9f6-26d0-45d6-8024-4c946fc83c20.JPG)

 
    
9. Class Wise Accuracy score:
    -------------
    **CIFAR10.ipynb:**
   
          car : 96.43
          
          bird : 72.73
          
          cat : 67.65
          
          deer : 77.78
          
          dog : 78.79
          
          frog : 88.89
          
          horse : 92.0
          
          ship : 93.75
          
          truck : 87.18




#### Improvement Plan : To get it run for more epochs and need to either tweak architecture or use regularizations to improve the performance.






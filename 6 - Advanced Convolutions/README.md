  # CIFAR-10 (CIFAR10_model.ipynb/CIFAR10_62Epochs.ipynb)
                                                        
                                                        
                                                        
                                                  
  **Session-6: Advanced Convolutions** : Describing about normal convolutions and its expensive computations, introduction of dilation convolution and depthwise convolutions and group convolutions and their expensive computation compared to other type of convolution and their usage and their methodologies.
                         
                                                      

Assignment: 

Fix the network: https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw

   <img src="https://user-images.githubusercontent.com/60026221/217057292-3ec2cb8a-79f4-4494-ab39-33c8d1116037.JPG" width=40% height=40%>


### The repo contains 
      
      A. main.py where the necessary training & testing happens and returns scores
      
      B. model.py which has the architecture built using PyTorch

      C. utils.py which has useful functions like normalizing, dataloading, briefly called as helper functions and also returns class accuracy

      D. CIFAR10_model.ipynb and CIFAR10_62Epochs.ipynb are of same architecture, they just ran for different number of epochs. The CIFAR10_model which was planned for 84 and CIFAR10_62Epochs was planned for 62 Epochs.
      
      E. less_weighted-model.ipynb is a notebook which contains all features in it which doesnt import files like main.py, utils, model and architecture consists of 191K Parameters.

   
   
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
       
       * Block1: [Conv1(3X3X16) with padding + Conv2(3x3x32) + Conv3(3x3x32)]
       * Block2: [Conv4(5x5x128 i,e,kernel incl. of dilation2) with dilated convolution (Dilation=2) + Conv5(3x3x64) + Conv6(3x3x32) with stride of 2 and padding]
       * Block3: [[Conv7(3x3x128)+Conv8(1x1x256] a depthwise convolution + Conv9(3x3x32)]
       * BLock4: [Conv10(3x3x32) + conv11(1x1x16) + conv12(3x3x10)]
       * GAP LAYER : Kernel_size = 3
       
    * There are total of **187,418 Parameters ~ 187K**
    
4. Ran For 84 Epochs in CIFAR10_model.ipynb.

6. Analysis: 

    * **less_weighted-model.ipynb:** With out augmentation reached training accuracy of 86% and val-accuracy of 77%, as it seems to be ok but not good. There is a need to avoid overfitting and add variance to data. Hence, we have procedded with augmenting the images using albumentations library and this notebook architecture is  4 convolution blocks are used with total of 12 layers in-it. The architecture also contains the dilation convolution (1-layer) and depthwise (1-layer) along with pointwise(1-layer). It summarizes as 9 convolution layers + 1 Dilation layer + 1 Depthwise seperable convolutions are used and they made up the number of **parameters as: 1,91,514**
     
    * **CIFAR10_model.ipynb:** With augmentation, it performed quite well but has to be improved more. 
    
    * **CIFAR10_62.ipynb:** With augmentation, it ran for all determined 62 epochs and gave 84 as highest val-accuracy:
    
        EPOCH: 58/62
        
        Loss=0.6362294554710388 Batch_id=390 train-Accuracy=76.80: 100%|██████████| 391/391 [00:20<00:00, 19.35it/s]
        
        Test set: Average loss: 0.0037, val-Accuracy: 8423/10000 (84.23%)
    
7. Accuracies: 
   
    * Model:
    ---------
      **CIFAR10_model.ipynb**
      
        EPOCH: 45/84

        Loss=0.7905847430229187 Batch_id=390 train-Accuracy=76.01: 100%|██████████| 391/391 [00:21<00:00, 18.23it/s]

        Test set: Average loss: 0.0038, val-Accuracy: 8328/10000 (83.28%)
      
      **CIFAR10_62Epochs.ipynb** 
      
        EPOCH: 58/62
        
        Loss=0.6362294554710388 Batch_id=390 train-Accuracy=76.80: 100%|██████████| 391/391 [00:20<00:00, 19.35it/s]
        
        Test set: Average loss: 0.0037, val-Accuracy: 8423/10000 (84.23%)
 
    
    * Class Wise: 
    -------------
    **CIFAR10_62Epochs.ipynb:**
    
        plane : 93.1%

        car : 96.43%

        bird : 63.64%

        cat : 79.41%

        deer : 70.37%

        dog : 66.67% 

        frog : 91.67% 

        horse : 88.0%

        ship : 93.75% 

        truck : 87.18% 
    




#### Pending : To get it run for more epochs and need to either tweak architecture or use regularizations to improve the performance.






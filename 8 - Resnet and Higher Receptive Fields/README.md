## ResNet & Receptive Fields
----------------------------

**Session-8: Resnet & Receptive Fields** - Describing about the architecture of architecture's like Resnet, Inception, VGG & the importance of receptive fields, impact of larger receptive fields, how the does the image & its variations occur in larger receptive field. How useful is the RF parameter in CNN's. Usage of One-Cycle-Policy(OCP) which is a kind of cyclic learning rate(CLR) but it only goes up and down once, whereas CLR continues to oscillate to & fro.

Assignment: 
----------
 ![S8_Assignment](https://user-images.githubusercontent.com/60026221/225386747-bd7194b0-35d9-44da-bfa8-626050e7b7eb.JPG)



### The Repo contains 

         resnet_clr.ipynb: A Jupyter notebook, which executes the resnet-18 layered implementation with the help of OCP to find lr.
                           The files are being imported from repo:  https://github.com/kishkath/CIFAR10-OCP
                           The complete implementation is based on importing files and classes and its methods. 
                             
      
      
* The model runs for 24 EPOCHS and training-accuracy of 93% and validation-accuracy of 88%.

         EPOCH: 23
         Current Learning Rate:  0.01678010204081637
         Loss=1.5123331546783447 Batch_id=97 train-Accuracy=93.73: 100%|██████████| 98/98 [00:21<00:00,  4.55it/s]
         updated Learning Rate:  0.0001301020408163156

         Test set: Average loss: 0.0032, val-Accuracy: 8857/10000 (88.57%)



Plots: 
------

**Performance Curves:**

![Acc_S8](https://user-images.githubusercontent.com/60026221/225387494-bf5e7236-629c-4a4a-b4e2-d2f50385d839.JPG)


![loss_s8](https://user-images.githubusercontent.com/60026221/225387502-3e00955a-6560-4375-a752-641fba041a21.JPG)

**Mis-Classifications:**

![mis_clas_S8](https://user-images.githubusercontent.com/60026221/225387662-6f9d339f-8237-478c-8fa3-f28a315d5f9d.JPG)

**Grad-Cam Visuals:**

![gradCAM_S8](https://user-images.githubusercontent.com/60026221/225387767-cd77aa71-e224-4962-a469-ab541b705645.JPG)



 
      




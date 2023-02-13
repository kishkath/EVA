## ResNet (CIFAR10_ResNet18)
------------------------------

**Session-7: Advanced Training Concepts** - Describing about model explainability via usage of GRADCAM, and usage of learning rates and its values,
different optimizers and its comparisons and its counter-parts.

Assignment: 



### The Repo contains 

      A. CIFAR10_ResNet18.ipynb: A Jupyter notebook, which executes the resnet-18 layered implementation. 
                                 The files are being imported from repo:  https://github.com/kishkath/CIFAR10
                                 The complete implementation is based on importing files and classes and its methods. 
                                 
      B. MiSCLASSIFIED_IMAGES  : Image which are mis-classified.
      
      C. GRADCAM Images.pdf    : Images are plotted and cut and pasted in document.
      
      
* The model runs for 20 EPOCHS and got the training-accuracy of 96% & validation-accuracy of 83%.
 
      EPOCH: 19
      --------
      Loss=0.34115737676620483 Batch_id=390 train-Accuracy=96.82: 100%|██████████| 391/391 [00:44<00:00,  8.72it/s] 
      Test set: Average loss: 0.0059, val-Accuracy: 8300/10000 (83.00%)
      
* Plots: 


![loss](https://user-images.githubusercontent.com/60026221/218268512-59d7fb99-8371-418d-99f9-cbc2e9c1bdd6.JPG)


![acc](https://user-images.githubusercontent.com/60026221/218268524-3e1406d1-ad6c-491c-a217-4c5b8484ac8a.JPG)


* Few of Mis-classified Images & its model explainability with GRADCAM.


![misclasfied](https://user-images.githubusercontent.com/60026221/218516617-65f3b2dc-7f4f-4fde-956d-ea2ea0d6fec7.png)


- GRADCAM VISUALS for 2 mis-classifications which was of layer2, which means as the model is still to be able to see the whole image the visualization of gradcam in layer-2 anyways will be clumsy, for deeper and neat prediction we can try GRADCAM on the final layers.:

![gradCAM1](https://user-images.githubusercontent.com/60026221/218516910-7ef94d4b-b6d0-4a85-b6fc-1f946858dc4d.JPG)




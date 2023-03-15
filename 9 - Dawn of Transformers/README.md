## Transformer
----------------------------

**Session-9: Dawn of Transformer** - It is  one of the most hot trending Architecture in AI industry where all the Deep learning things
(i.e,) Image, Video, Audio, Text gonna work with a single architecture making them a "Multi-Modal-Network". 
* "Attention" Please: This session is all about introducing transformer to a beginner where it describes about the basic steps of building the network. 
Session Contains description of basic transformer architecture, pre-foundation 
things such as full-context(detecting person in image), 
selective context(Only we can detect whatever required like only eyes of person in an image containing a person),
Embedding (numerical representation with context), Patch embeddings, encoders, decoders, MLP, Key, Query, Value and the most important "Attention". 

https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1


Assignment: 
----------

![S9](https://user-images.githubusercontent.com/60026221/225396890-8122d246-6e54-4638-bf46-b006a34a9561.JPG)




### The Repo contains 

         resnet_clr.ipynb: A Jupyter notebook, which executes the resnet-18 layered implementation with the help of OCP to find lr.
                           The files are being imported from repo:  https://github.com/kishkath/CIFAR10-OCP
                           The complete implementation is based on importing files and classes and its methods. 
                             
      
      
* The model runs for 24 EPOCHS and training-accuracy of 93% and validation-accuracy of 88%.

        EPOCH: 19
        Current Learning Rate:  0.01678010204081637
        Loss=1.5123331546783447 Batch_id=97 train-Accuracy=93.73: 100%|██████████| 98/98 [00:21<00:00,  4.55it/s]
        updated Learning Rate:  0.0001301020408163156

        Test set: Average loss: 0.0032, val-Accuracy: 8857/10000 (88.57%)
        


Plots: 
------

**Performance Curves:**

![Acc_S9](https://user-images.githubusercontent.com/60026221/225396868-2a5b2778-5a58-4e34-bba4-823f2c7b9206.JPG)

![loss_S9](https://user-images.githubusercontent.com/60026221/225396880-8d3d95b9-9e34-4a79-89fa-4742a601ea01.JPG)

**Mis-Classifications:**

![mis_class_S9](https://user-images.githubusercontent.com/60026221/225396884-6cdf2047-772e-4dd1-9d63-68bacb1dd052.JPG)

**Grad-Cam Visuals:**

![gradCAM_S9](https://user-images.githubusercontent.com/60026221/225396874-6ba51082-64cb-49ba-afe2-cb67b272eefa.JPG)



 
      




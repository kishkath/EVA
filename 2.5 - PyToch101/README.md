**Neural Network**
==================

#### Description

    Know About Neural Network: A Neural Network is a combination of several layers which together combinely called as "Network" and as they contain small circles called as neurons which is used for storage/computation/transfer of data. These neurons are connected by a line/wire/thread/medium(information passer) where weights are placed and bias at the entire layer is used as it is historically defined. The first layer of netowrk is called Input Layer & last is called Output Layer and the rest in between are called as, hidden layers (we really dont know what's happening) 😮‍💨.
  
 * So, here we are using this type of network and trying to predict the handwritten digits which are stored in form of pixel values in a 2D-Matrix of shape (28,28)
 * as well as the random number that is generated.
 * Visualized network is shown below.
 
**Tech-stack**
--------------
1. Python - for coding.
2. Pytorch - used as framework for building neural network


**Statement**
=============
* Building a Neural network that will take 
  - 2 inputs: 

       ***input1***: an image from the MNIST dataset (say 5)
       
       ***input2***: a random number between 0 and 9, (say 7)
       
  - 2 outputs:
  
      ***output1***: "number" that was represented by the MNIST image (predict 5)
      
      ***output2***: "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
      

    
 * Visualization    
      
      ![neuralNetwork](https://user-images.githubusercontent.com/60026221/211019570-6e851cab-eff1-4742-8502-48968e719ede.jpg)


* Training Logs:

**Trained for 300 Epochs with batch_size=128**

###### Logs
    epoch 0 loss: 3764.1526532173157
    epoch 15 loss: 3764.3100872039795
    epoch 30 loss: 3764.1765875816345
    epoch 45 loss: 3764.03285074234
    epoch 60 loss: 3764.1903777122498
    epoch 75 loss: 3764.077751636505
    epoch 90 loss: 3764.096091270447
    epoch 105 loss: 3764.1592144966125
    epoch 120 loss: 3764.1476073265076
    epoch 135 loss: 3764.228371143341
    epoch 150 loss: 3764.3420395851135
    epoch 165 loss: 3764.1188020706177
    epoch 180 loss: 3764.186189174652
    epoch 195 loss: 3764.2435116767883
    epoch 210 loss: 3764.1349654197693
    epoch 225 loss: 3764.2512426376343
    epoch 240 loss: 3764.0984206199646
    epoch 255 loss: 3764.1906995773315
    epoch 270 loss: 3764.1153264045715
    epoch 285 loss: 3764.179489135742

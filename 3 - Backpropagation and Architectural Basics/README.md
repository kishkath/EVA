
# Part-1: 
--------
### Figure: Neural Network where the parameters such as weights (noted as w,..) & neurons (the circles that are visible) & layers

       Note: For ease usage, can view the .jpg files['forward.JPG','backward.JPG','learning1.JPG','learning2.JPG'] attached on behalf of excel file.
--------------------

  ![neuralNtw](https://user-images.githubusercontent.com/60026221/212375211-a5ad76dc-50a7-48d6-98f1-b88aa8d2b6e3.JPG)



The Excel file contains:

1. Forward Propogation: The network & lines moves forward and in this propogation,the values are calculated(Weighted sum of inputs) & stored in neurons and fed forward torward next layer.

  ![fedforward](https://user-images.githubusercontent.com/60026221/212379897-13671ed2-917b-44f1-9102-194d3a728af2.jpg)

2. Backward Propogation: The network starts from the output layer, once the epoch is ran it starts computing differentiation values of loss function w.r.t, the parameters ahead from the last layer (ex: The output layer has parameters such as Activation(h1) & Activation(h2) & Weights). **These differentiation values are calculated and are multiplied with learning rate (defined as below) and are subtracted from original weight value, so that the weights decreases called model learning and help in reaching minimum loss at the output.**

  ![backprop](https://user-images.githubusercontent.com/60026221/212380182-97589d99-1567-4494-920f-4fe7d6874121.jpg)

3. Calculation of weights & other parameters with different learning rates.



# Part-2: 
--------

* AIM: Modifying/Tuning CNN Architecture to get 99.4% Validation accuracy using less than 20K parameters and of less than 20 Epochs.

Parameters used: 14,666

Logs: 



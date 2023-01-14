
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

- Parameters used: 14,666

- Best Validation-Accuracy: 99.20%

- 5 Layered network with 4 convolution layers and a fully connected layer 

- Logs: 
       Epoch:  1
       --------------
       loss=0.14157654345035553 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.12it/s]
       Test set: Average loss: 0.0670, Accuracy: 9785/10000 (97.85%)

       Epoch:  2
       --------------
       loss=0.06746566295623779 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.93it/s]
       Test set: Average loss: 0.0477, Accuracy: 9830/10000 (98.30%)

       Epoch:  3
       --------------
       loss=0.0667879655957222 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.63it/s]
       Test set: Average loss: 0.0407, Accuracy: 9852/10000 (98.52%)

       Epoch:  4
       --------------
       loss=0.05491636320948601 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.16it/s]
       Test set: Average loss: 0.0384, Accuracy: 9871/10000 (98.71%)

       Epoch:  5
       --------------
       loss=0.031515952199697495 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.30it/s]
       Test set: Average loss: 0.0319, Accuracy: 9885/10000 (98.85%)

       Epoch:  6
       --------------
       loss=0.040687981992959976 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.01it/s]
       Test set: Average loss: 0.0329, Accuracy: 9878/10000 (98.78%)

       Epoch:  7
       --------------
       loss=0.08007332682609558 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.48it/s]
       Test set: Average loss: 0.0320, Accuracy: 9884/10000 (98.84%)

       Epoch:  8
       --------------
       loss=0.023730916902422905 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.97it/s]
       Test set: Average loss: 0.0272, Accuracy: 9910/10000 (99.10%)

       Epoch:  9
       --------------
       loss=0.018825998529791832 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.66it/s]
       Test set: Average loss: 0.0260, Accuracy: 9909/10000 (99.09%)

       Epoch:  10
       --------------
       loss=0.01291016023606062 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.25it/s]
       Test set: Average loss: 0.0267, Accuracy: 9904/10000 (99.04%)

       Epoch:  11
       --------------
       loss=0.029386267066001892 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.89it/s]
       Test set: Average loss: 0.0261, Accuracy: 9905/10000 (99.05%)

       Epoch:  12
       --------------
       loss=0.014224191196262836 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.57it/s]
       Test set: Average loss: 0.0236, Accuracy: 9920/10000 (99.20%)

       Epoch:  13
       --------------
       loss=0.057570770382881165 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.39it/s]
       Test set: Average loss: 0.0244, Accuracy: 9913/10000 (99.13%)

       Epoch:  14
       --------------
       loss=0.006078090984374285 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.84it/s]
       Test set: Average loss: 0.0261, Accuracy: 9909/10000 (99.09%)

       Epoch:  15
       --------------
       loss=0.15223008394241333 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.88it/s]
       Test set: Average loss: 0.0246, Accuracy: 9913/10000 (99.13%)

       Epoch:  16
       --------------
       loss=0.0050852056592702866 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.56it/s]
       Test set: Average loss: 0.0257, Accuracy: 9908/10000 (99.08%)

       Epoch:  17
       --------------
       loss=0.012222692370414734 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.07it/s]
       Test set: Average loss: 0.0234, Accuracy: 9913/10000 (99.13%)

       Epoch:  18
       --------------
       loss=0.051050424575805664 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.60it/s]
       Test set: Average loss: 0.0229, Accuracy: 9920/10000 (99.20%)

       Epoch:  19
       --------------
       loss=0.018453702330589294 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.36it/s]
       Test set: Average loss: 0.0247, Accuracy: 9914/10000 (99.14%)




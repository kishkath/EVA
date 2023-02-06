** Session-6: Advanced Convolutions** 

Assignment: 

Fix the network: https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw

![assignment](https://user-images.githubusercontent.com/60026221/217057292-3ec2cb8a-79f4-4494-ab39-33c8d1116037.JPG)

* As of need, 4 convolution blocks are used with total of 12 layers in-it. The architecture also contains the dilation convolution (1-layer) and depthwise (1-layer) along with pointwise(1-layer). It summarizes as 9 convolution layers + 1 Dilation layer + 1 Depthwise seperable convolutions are used and they made up the number of **parameters as: 1,91,514**

* The repo contains 
   1. main.py where the necessary training & testing happens.
   2. model.py which has the architecture.
   3. utils.py which has useful functions like normalizing, albumentations, briefly called as helper functions.
   4. 2 Notebooks one with more 200K parameters which provided good result but have overfitting, which is trained without using any special features and other than less than 200K parameters which used special features.
   





# YoloV3

Opencv, a popular open-source library where one can use it for image processings. This opencv library has made a library with darknet which will include the YoloV3

through api. We can directly run the model from the api and use it for the testing data (The testing data should only contain the 80Classes of Yolov3, only those will

get predicted as it is a standard model). The results of the notebook are placed in 'opencv_inf_images'.

### predicted: 

      ![inference1](https://user-images.githubusercontent.com/60026221/230815812-5d26c5ac-e43e-45ba-a49c-929dbb060e2c.PNG)


# Pretrained-Yolov3 

YOLO(You Look Only Once), a cnn based architecture where the image will be seen only once through the iteration. The repo has been cloned from
  
https://github.com/theschoolofai?tab=repositories

and have made a forked version by making following changes that have been made to the respective files. 

The cloning and making the training has been done directly after cloning the repo with out making changes in the middle.

The following changes have been made to the forked version of repo:

1. Yolov3-custom.cfg: A config file in 'cfg' folder, which has all the necessary configurations to get the model learn from data and here
we modify the number of filters, as we will be using 3 anchors, we do get filters in the final layer as (4+1+4)*3 = 27 filters and need to modify the classes as number of classes we are using. 

       Line No: 636-643, 722-729, 809-816 

2. utils.py : A python module in 'utils' folder where we convert tensors to long() tensors & also modify them to be with gpu instead of cpu.

       Line No: After 394 :  b,a,gj,gi = b.long(),a.long(),gj.long(),gi.long()
       Line No: Above 475 :  t = t.to(targets.device), a = a.to(targets.device) 
       Line No: 877       :  type casting of float to int.

3. train.py : A python file which will be used for training purpose.

       Line No: 82 : Modifying the access of value of dictionary for test_path, from 'valid' to 'test'
       Line No: 195: Setting rect = False for testloader set.

4. test.py  : For testing/validation purposes.

       Line No: 55 : Same as change in train.py

### classes used: 

       1. A cricket 'bat' 
       2. Cricket 'helmet'
       3. 'smart_watch' 
       4. 'umpire'

### First Iteration: 

1. Collected around 50 images for each class which makes up of total 202 images for all 4 classes.
2. Trained for 300 epochs.
3. The videos are tested which are provided in "input_videos_run1"
4. The output videos are provided in "output_videos_run1" 

### Predicted: 

    ![inference5](https://user-images.githubusercontent.com/60026221/230815972-f8d06708-fdfe-4da7-af71-a6c7de542a80.PNG)


Analysis: As all the 4 class images are picked from 'google', each image consists of only a single object (for few images, there exists helmet, bat in a single image). But our testing data (i,e,,) output videos are of the real videos of cricket match, and the model performed good as per data. As from the videos, it is not been able to predict bat due to its movement which is very quick while a person is batting. Helmet and umpire has better predictions. The following iteration has to be made by adding few more images with the real-time images. So, to make it better have taken the output videos of run1 and extracted the frames from it and has trained run2 on those images along with addition some more frames of multiple videos.


**Need to add run2 Notebook & results**

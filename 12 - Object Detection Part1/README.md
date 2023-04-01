# part 2 changes 

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

classes used: 

1. A cricket 'bat' 
2. Cricket 'helmet'
3. 'smart_watch' 
4. 'umpire'

### First Iteration: 

1. Collected around 50 images for each class which makes up of total 202 images for all 4 classes.
2. 

**Session13: UNets & AutoEncoders**


Strided Conv of Encoder +  Transpose Conv of Decoder: 

train: bce: 0.314808, dice: 0.198193, loss: 0.256501
LR 0.0001
val: bce: 0.288205, dice: 0.192803, loss: 0.240504
3m 27s
Best val loss: 0.239968


Max Pooling of Encoder + Transpose Conv of Decoder : 

train: bce: 0.149071, dice: 0.087116, loss: 0.118094
LR 1e-05
val: bce: 0.156856, dice: 0.095543, loss: 0.126199
3m 24s
Best val loss: 0.125458

Strided Conv of Encoder + UpSample: 

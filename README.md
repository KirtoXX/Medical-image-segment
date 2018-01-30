# Segment network architecture
refernence paper:<p>
《The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation》  <p>
   Simon Jégou, Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua Bengio<p>
https://arxiv.org/abs/1611.09326  <p>
![image](https://github.com/KirtoXX/segment/blob/master/20170817161456238.png)
# Mychange
input size:512x512    <p>
use 7x7 conv and 2x2 stride down sample to 256x256  <p>
use 2x2 maxpooling down sample to 128x128 <p>
output size:128x128，interpolation—>512x512    <p>

| Layer         | feature map   | filter| Conv    |DB size |
| ------------- |:-------------:| -----:|--------:|-------:|
| Input         | 512x512       | 1     |  NULL   |
| Conv0         | 256x256       |  48   |   7x7   | 
| Maxpooling    | 128x128       |  48   |   2x2   | 
| DB1           | 128x128       |  120  |   3x3   | 3      | 
| TD1           | 64x64         |  120  |   1x1   | 
| DB2           | 64x64         |  240  |   3x3   | 5      |
| TD2           | 32x32         |  240  |   1x1   | 
| DB3           | 32x32         |  456  |   3x3   | 7      |
| TD3           | 16x16         |  456  |   1x1   | 
| Center        | 16x16         |  672  |   3x3   | 9      |
| TU5           | 32x32         |  456  |   1x1   |
| C5(TU5,DB3)   | 32x32         |  912  |   NULL  |
| DB5           | 32x32         |  456  |   3x3   |  7     |
| TU6           | 64x64         |  240  |   1x1   |
| C6(TU6,DB2)   | 64x64         |  480  |   NULL  |
| DB6           | 64x64         |  240  |   3x3   |  5     |
| TU7           | 128x128       |  120  |   1x1   |
| C7(TU7,DB1)   | 128x128       |  240  |   NULL  |
| DB7           | 128x128       |  120  |   3x3   |  3     |
| Output        | 128x128       |  1    |   1x1   |
<p><p>
layer:51  <p>
total size of network:26Mb    <p>

# Result
![image](https://github.com/KirtoXX/segment/blob/master/tiramasu56.png)
   
  
  
  
  
  
  


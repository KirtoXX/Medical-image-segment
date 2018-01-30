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

| net           | feature map   | filter|  size   |
| ------------- |:-------------:| -----:|--------:|
| Input         | 512x512       | 1     |  null   |
| conv0         | 256x256       |  48   |   7x7   | 
| maxpooling    | 128x128       |  48   |   2x2   | 
| DB            | 128x128       |  120  |   3x3   | 
| TD            | 64x64         |  120  |   1x1   | 
| DB            | 64x64         |  240  |   3x3   | 
| TD            | 32x32         |  240  |   1x1   | 
| DB            | 32x32         |  456  |   3x3   | 
| TD            | 16x16         |  456  |   1x1   | 
| DB            | 16x16         |  672  |   3x3   |

layer:51  <p>
total size of network:26Mb    <p>
# Result
![image](https://github.com/KirtoXX/segment/blob/master/tiramasu56.png)
   
  
  
  
  
  
  


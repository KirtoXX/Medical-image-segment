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

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

layer:51  <p>
total size of network:26Mb    <p>
# Result
![image](https://github.com/KirtoXX/segment/blob/master/tiramasu56.png)
   
  
  
  
  
  
  


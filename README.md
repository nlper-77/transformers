# transformer及其变体

## MASK
![MASK](img\transofrmer_mask.jpg)
## Multi-head Attention
![multi-head](img\multi-head.png)
第一种方式，code-1就是这种实现方法。

第三种方式是，参考1 中的实现方式，直接将x赋值给query, key, value, 将768维reshape成12x64, 然后定义一个q_w(64x64), k_w(64x64), v_w(64x64)，所以参数量就是64x64x3 + 12*64*768(全连接输出层)
## Papers
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
## Codes
1. https://github.com/voidmagic/Beaver/tree/master/beaver/model
## 参考
1. [矩阵视角下的Transformer详解（附代码）](https://mp.weixin.qq.com/s/ZllvtpGfkLrcUBKZDtdoTA)
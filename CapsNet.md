### CapsNet.py by [InnerPeace Wu](https://github.com/InnerPeace-Wu/CapsNet-tensorflow/)

여기서는 [Dynamic Routing between Capsules](https://arxiv.org/abs/1710.09829) 논문을 구현한 코드를 분석한다.
Python 문법에 대해서도 잘 모르는 것이 많기 때문에 line-by-line으로 무슨 내용인지 모두 이해할 수 있도록 한다.

```python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```

__future__는 python 2와 python 3의 차이를 메꾸기 위해서 사용하는 모듈이다. 예를 들어 print 같은 경우 python 2에서는 키워드지만,
python 3에서는 함수이기 때문에 어떻게 쓰느냐에 따라 결과가 다르게 나온다 [(참조)].(http://www.hanbit.co.kr/network/category/category_view.html?cms_code=CMS9324226566)
__future__를 사용해주면 python 2,3 어디에서 쓰든지 같은 결과를 얻을 수 있다.

```python3
import time
import os
import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib import slim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import squash, imshow_noax, tweak_matrix
from config import cfg
```

import 하는 모듈과 함수 중에 뭐하는 것인지 모르는게 많지만, 여기서 찾지 않고 실제 그 함수를 사용할 때 찾도록 하겠다.

```python3
class CapsNet(object):
    def __init__(self):
        """initial class with mnist dataset"""
        # keep tracking of the dimension of feature maps
        self._dim = 28
        # store number of capsules of each capsule layer
        # the conv1-layer has 0 capsules
        self._num_caps = [0]
        # set for counting
        self._count = 0
        # set up placeholder of input data and labels
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y_ = tf.placeholder(tf.float32, [None, 10])
        # set up initializer for weights and bias
        self._w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self._b_initializer = tf.zeros_initializer()

```

CapsNet이라는 클래스 및 멤버변수 정의.

```python3
    def _capsule(self, input, i_c, o_c, idx):
        """
        compute a capsule,
        conv op with kernel: 9x9, stride: 2,
        padding: VALID, output channels: 8 per capsule.
        As described in the paper.
        :arg
            input: input for computing capsule, shape: [None, w, h, c]
            i_c: input channels
            o_c: output channels
            idx: index of the capsule about to create

        :return
            capsule: computed capsule
        """
        with tf.variable_scope('cap_' + str(idx)):
            w = tf.get_variable('w', shape=[9, 9, i_c, o_c], dtype=tf.float32)
            cap = tf.nn.conv2d(input, w, [1, 2, 2, 1],
                               padding='VALID', name='cap_conv')
            if cfg.USE_BIAS:
                b = tf.get_variable('b', shape=[o_c, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                cap = cap + b
            # cap with shape [None, 6, 6, 8] for mnist dataset

            # Note: use "squash" as its non-linearity.
            capsule = squash(cap)
            # capsule with shape: [None, 6, 6, 8]
            # expand the dimensions to [None, 1, 6, 6, 8] for following concat
            capsule = tf.expand_dims(capsule, axis=1)

            # return capsule with shape [None, 1, 6, 6, 8]
            return capsule
```

먼저, capsule에 대한 설명을 해 놓은 블로그가 많은데, [이곳](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)이 좋은 것 같다.
w, cap 등의 변수 정의를 보면, capsule에서 사용하는 convolution은 9x9 커널을 사용하고, 엣지를 늘리지 않는 'VALID' 타입 패딩을 사용하며,
stride는 가로, 세로 모두 2 픽셀임을 알 수 있다. 

결과가 [None, 6, 6, 8]인 이유는, 원래 28x28인 MNIST 이미지를 capsule layer 앞에 있는 convolutional layer에서 20x20으로 만들기 때문이다.




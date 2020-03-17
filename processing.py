import torch
import torch.nn as nn

video=torch.ones((1,1,200,134))
print('input size', video.size())

wide_filter = (10,5)
tall_filter = (5,10)
stride = 1

wide_conv = nn.Conv2d(1,1, wide_filter, stride)
tall_conv = nn.Conv2d(1,1, tall_filter, stride)

feat1 = wide_conv(video)
feat2 = tall_conv(video)
print('wide feature map', feat1.size(), '\ntall feature map', feat2.size())


for i,j in zip(feat1.size(), feat2.size()):
    print(i,j)

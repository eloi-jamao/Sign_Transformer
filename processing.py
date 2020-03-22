import torch
import torch.nn as nn

video=torch.ones((1,1,200,134))
print('input size', video.size())

wide_filter = (12,6)
tall_filter = (6,12)

wide_conv = nn.Conv2d(1,1, wide_filter, 1)
tall_conv = nn.Conv2d(1,1, tall_filter, 1)

feat1 = wide_conv(video)
feat2 = tall_conv(video)
print('wide feature map', feat1.size(), '\ntall feature map', feat2.size())

pady, padx = abs(feat1.size()[-1] - feat2.size()[-1]), abs(feat1.size()[-2] - feat2.size()[-2])
pad1 = nn.ZeroPad2d((0, 0, pady//2, pady//2))
pad2 = nn.ZeroPad2d((padx//2, padx//2, 0, 0))

feat1 = pad1(feat1)
feat2 = pad2(feat2)
print('paded wide feature map ', feat1.size(), '\npaded tall feature map', feat2.size())

#same size, now we can concatenate

feat = torch.cat((feat1,feat2), dim=-3)
print('concatenated size', feat.size())

#Final convolution taking 2 chanels and returning 1

final_conv = nn.Conv2d(2,1, (5,5), 2)

out = final_conv(feat)
print('output size', out.size())
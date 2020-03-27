import torch.nn as nn
import model
import feature_extraction.resnets_3d.resnet as rn

# TODO
class Seq2Seq(nn.Module):
    def __init__(self, opt_t, opt_r):
        super(Seq2Seq, self).__init__()

        resnet = rn.resnet34(num_classes=opt_r.n_classes, shortcut_type=opt_r.resnet_shortcut,
                             sample_size=opt_r.sample_size,
                             sample_duration=opt_r.sample_duration)
        self.extractor = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3, resnet.layer4)
        self.linear = nn.Sequential(nn.Linear(512, opt_t.ntoken), nn.ReLU(inplace=True))
        self.transformer = model.TransformerModel(opt_t.ntoken, opt_t.ninp, opt_t.nhead, opt_t.nhid, opt_t.nlayers,
                                                  opt_t.vocab_size, opt_t.max_seq, opt_t.dropout)

    def forward(self, x):
        y = self.resnet(x)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        y = self.transformer(y)
        return y

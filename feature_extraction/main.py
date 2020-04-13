import os
import json
import torch
from feature_extraction.resnets_3d import *
import feature_extraction.resnets_3d.spatial_transforms as st
from feature_extraction.resnets_3d.model import generate_model
from feature_extraction.resnets_3d.mean import get_mean, get_std
from feature_extraction.resnets_3d.opts import parse_opts


if __name__ == '__main__':

    opt = parse_opts()

    opt.root_path = os.getcwd()
    # opt.pretrain_path = "model/resnet-34-kinetics.pth"
    opt.model_depth = 34


    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    # with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #     json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    print(model.parameters())
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = True


# base_dir = "/gdrive/My Drive/slt/testData"
#
# model = resnet50(sample_size=sample_size,
#                  sample_duration=sample_duration,
#                  last_fc=False)
# print(model)
#
# """model = resnet34(sample_size=112,
#                  sample_duration=16,
#                  last_fc=True)"""
#
# model_data = torch.load("/gdrive/My Drive/slt/models/resnet-50-kinetics.pth")
#
# state_dict = {}
# for key, value in model_data['state_dict'].items():
#   key = key.replace("module.", "")
#   state_dict[key] = value
# model.load_state_dict(state_dict)
#
# #model.load_state_dict(model_data['state_dict'])
# model.eval()
#
# for d in os.listdir(base_dir):
#   #print(os.path.join(base_dir, d))
#   result = extract_features(os.path.join(base_dir, d))
#   break



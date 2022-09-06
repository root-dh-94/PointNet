from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='../pretrained_networks/latest_segmentation.pt', help='model path')
parser.add_argument('--idx', type=int, default=1, help='model index')
parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help='dataset path')
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
#classifier = PointNetDenseCls(num_classes= state_dict['model']['mlp4.weight'].size()[0], feature_transform=True)
classifier = PointNetDenseCls(num_classes= 4, feature_transform=True)
classifier.cuda()

classifier.load_state_dict(state_dict['model'])
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
point = point.cuda()
pred, _, _, indeces= classifier(point)
pred_choice = pred.data.max(dim=1)[1]
print(pred_choice)
print(pred_choice.size())
#indeces = indeces.squeeze(0).squeeze(1).cpu().tolist()
#critical_points= point_np[indeces]
pred_color = cmap[pred_choice.cpu().numpy()[0], :]
#showpoints(critical_points, None,None)
print(pred_color.shape)
showpoints(point_np, gt, pred_color)

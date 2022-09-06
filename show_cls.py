from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F
#Uncomment line below to run critical point visualization
#from show3d_balls import showpoints
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--idx', type=int, default=1, help='index  of crticial visualization')


opt = parser.parse_args()
print(opt)
idx = opt.idx
test_dataset = ShapeNetDataset(
    #root='../shapenet_data/shapenetcore_partanno_segmentation_benchmark_v0',
    root='../shapenetcore_partanno_segmentation_benchmark_v0',
    classification=True,
    split='test',
    npoints=2500)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

classifier = PointNetCls(num_classes=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model)['model'])
classifier.eval()


total_preds = []
total_targets = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        #TODO
        # calculate average classification accuracy
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        preds, _, _,_= classifier(points)
        pred_labels = torch.max(preds, dim= 1)[1]

        total_preds = np.concatenate([total_preds, pred_labels.cpu().numpy()])
        total_targets = np.concatenate([total_targets, target.cpu().numpy()])
        a = 0
    accuracy = 100 * (total_targets == total_preds).sum() / len(test_dataset)
    print('Accuracy = {:.2f}%'.format(accuracy))

points, target = test_dataset[idx]
point_np = points.numpy()

points = points.transpose(1, 0).contiguous()
points = Variable(points.view(1, points.size()[0], points.size()[1]))
points = points.cuda()
preds, _, _, indeces = classifier(points)
indeces = indeces.squeeze(0).squeeze(1).cpu().tolist()
critical_points = point_np[indeces]
#uncomment line below to run critical point visualization
#showpoints(critical_points, None, None)



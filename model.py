# -- coding: utf-8 --

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)


class HyConvE(BaseClass):

    def __init__(self, dataset, emb_dim, emb_dim1):
        super(HyConvE, self).__init__()

        self.dataset = dataset
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.lmbda = 0.15
        self.ent_embeddings = nn.Embedding(self.dataset.num_ent, self.emb_dim)
        self.rel_embeddings = nn.Embedding(self.dataset.num_rel, self.emb_dim)



        self.conv_layer_2 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 3))
        self.conv_layer_3 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 4))
        self.conv_layer_4 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 5))
        self.conv_layer_5 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 6))
        self.conv_layer_6 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 7))
        self.conv_layer_7 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 8))
        self.conv_layer_8 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 9))
        self.conv_layer_9 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 1, 10))
        self.fc_pos = nn.Linear(in_features=self.dataset.arity_lst[-1], out_features=9)
        self.fc_rel_2 = nn.Linear(in_features=self.emb_dim, out_features=3)
        self.pool = torch.nn.MaxPool3d((2, 1, 1))
        self.pool1d = torch.nn.MaxPool2d((1, 2))

        self.inp_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)
        self.dropout_3d = nn.Dropout(0.2)
        self.dropout_2d = nn.Dropout(0.2)
        self.nonlinear = nn.ReLU()
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 2
        self.conv_size_1d = (self.emb_dim) * 3 // 2
        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=1)
        self.fc_2 = nn.Linear(in_features=2*self.conv_size_1d, out_features=self.conv_size)
        self.fc_3 = nn.Linear(in_features=3*self.conv_size_1d, out_features=self.conv_size)
        self.fc_4 = nn.Linear(in_features=4*self.conv_size_1d, out_features=self.conv_size)
        self.fc_5 = nn.Linear(in_features=5*self.conv_size_1d, out_features=self.conv_size)
        self.fc_6 = nn.Linear(in_features=6*self.conv_size_1d, out_features=self.conv_size)
        self.fc_7 = nn.Linear(in_features=7*self.conv_size_1d, out_features=self.conv_size)
        self.fc_8 = nn.Linear(in_features=8*self.conv_size_1d, out_features=self.conv_size)
        self.fc_9 = nn.Linear(in_features=9*self.conv_size_1d, out_features=self.conv_size)

        self.bn1 = nn.BatchNorm3d(num_features=1)
        self.bn2 = nn.BatchNorm3d(num_features=4)
        self.bn3 = nn.BatchNorm2d(num_features=1)
        self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        self.criterion = nn.Softplus()

        # 初始化 embeddings 以及卷积层、全连接层的参数
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_2.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_3.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_4.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_5.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_6.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_7.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_8.weight.data)
        nn.init.xavier_uniform_(self.conv_layer_9.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.fc_rel_2.weight.data)
        nn.init.xavier_uniform_(self.fc_2.weight.data)
        nn.init.xavier_uniform_(self.fc_3.weight.data)
        nn.init.xavier_uniform_(self.fc_4.weight.data)
        nn.init.xavier_uniform_(self.fc_5.weight.data)
        nn.init.xavier_uniform_(self.fc_6.weight.data)
        nn.init.xavier_uniform_(self.fc_7.weight.data)
        nn.init.xavier_uniform_(self.fc_8.weight.data)
        nn.init.xavier_uniform_(self.fc_9.weight.data)



    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def conv3d_process(self, batch):
        if len(batch) == 3:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_2(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 4:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_3(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)


        if len(batch) == 5:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_4(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 6:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = batch[5].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_5(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 7:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = batch[5].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = batch[6].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_6(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 8:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = batch[5].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = batch[6].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = batch[7].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_7(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 9:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = batch[5].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = batch[6].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = batch[7].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e8 = batch[8].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_8(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 10:
            r = batch[0].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e1 = batch[1].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e2 = batch[2].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e3 = batch[3].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e4 = batch[4].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e5 = batch[5].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e6 = batch[6].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e7 = batch[7].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e8 = batch[8].view(-1, 1, self.emb_dim1, self.emb_dim2)
            e9 = batch[9].view(-1, 1, self.emb_dim1, self.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8, e9), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_9(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        x = x.view(-1, self.conv_size)

        x = self.dropout_3d(x)

        return x

    def convolve(self, e_emb, r_emb, pos):

        x = e_emb
        x = self.inp_drop(x)

        k1 = self.fc_rel_2(r_emb)
        k1 = k1.view(-1, 1, 3, 1, 1)
        k1 = k1.view(e_emb.size(0)*3, 1, 1, 1)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k1, groups=e_emb.size(0))


        one_hot_target = (pos == torch.arange(self.dataset.arity_lst[-1]).reshape(self.dataset.arity_lst[-1])).float().to(self.dataset.device)
        poses = one_hot_target.repeat(r_emb.shape[0]).view(-1, self.dataset.arity_lst[-1])
        one_hot_target.requires_grad = False
        poses.requires_grad = False

        k = self.fc_pos(poses)
        k = k.view(e_emb.size(0)*3, 3, 1, 1)
        x = F.conv2d(x, k, groups=e_emb.size(0), stride=1)
        x = x.view(e_emb.size(0), 1, 3, 1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def conv2d_process(self, batch):
        if len(batch) == 3:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)

            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2), dim=1)

            x = self.pool1d(x)


            x = x.view(e1.shape[0], -1)

            x = self.nonlinear(x)
            x = self.dropout(x)

            x = self.fc_2(x)

            return x

        if len(batch) == 4:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_3(x)

            return x

        if len(batch) == 5:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)

            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_4(x)

            return x

        if len(batch) == 6:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            e5 = batch[5].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)

            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5), dim=1)
            x = self.pool1d(x)


            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_5(x)

            return x

        if len(batch) == 7:
            r = batch[0].view(-1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            e5 = batch[5].view(-1, 1, 1, self.emb_dim)
            e6 = batch[6].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_6(x)
            return x

        if len(batch) == 8:
            r = batch[0].view(-1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            e5 = batch[5].view(-1, 1, 1, self.emb_dim)
            e6 = batch[6].view(-1, 1, 1, self.emb_dim)
            e7 = batch[7].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_7(x)
            return x


        if len(batch) == 9:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            e5 = batch[5].view(-1, 1, 1, self.emb_dim)
            e6 = batch[6].view(-1, 1, 1, self.emb_dim)
            e7 = batch[7].view(-1, 1, 1, self.emb_dim)
            e8 = batch[8].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3)
            conv_e8 = self.convolve(e8, r, 7).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_8(x)

            return x

        if len(batch) == 10:
            r = batch[0].view(-1, 1, 1, self.emb_dim)
            e1 = batch[1].view(-1, 1, 1, self.emb_dim)
            e2 = batch[2].view(-1, 1, 1, self.emb_dim)
            e3 = batch[3].view(-1, 1, 1, self.emb_dim)
            e4 = batch[4].view(-1, 1, 1, self.emb_dim)
            e5 = batch[5].view(-1, 1, 1, self.emb_dim)
            e6 = batch[6].view(-1, 1, 1, self.emb_dim)
            e7 = batch[7].view(-1, 1, 1, self.emb_dim)
            e8 = batch[8].view(-1, 1, 1, self.emb_dim)
            e9 = batch[9].view(-1, 1, 1, self.emb_dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3)
            conv_e8 = self.convolve(e8, r, 7).permute(0, 2, 1, 3)
            conv_e9 = self.convolve(e9, r, 8).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8, conv_e9), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_9(x)

            return x

    def forward(self, batch, labels):

        r = self.rel_embeddings(batch[:, 0])
        ents = self.ent_embeddings(batch[:, 1:])

        e1 = ents[:, 0]
        e2 = ents[:, 1]
        if batch.shape[1] == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)

            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2)
            for p in self.conv_layer_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_2.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)


            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 4:
            e3 = ents[:, 2]

            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2)

            for p in self.conv_layer_3.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_3.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)

            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2)

            for p in self.conv_layer_4.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_4.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2)

            for p in self.conv_layer_5.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_5.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[1] == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]


            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2)

            for p in self.conv_layer_6.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_6.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2)

            for p in self.conv_layer_7.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_7.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular


        if batch.shape[1] == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2) + torch.mean(e8 ** 2)

            for p in self.conv_layer_8.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_8.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[1] == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)

            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2) + torch.mean(e8 ** 2) + torch.mean(e9 ** 2)

            for p in self.conv_layer_9.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_9.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular



        return mean + regular

    def predict(self, test_batch):
        r = self.rel_embeddings(test_batch[:, 0])
        ents = self.ent_embeddings(test_batch[:, 1:])
        e1 = ents[:, 0]
        e2 = ents[:, 1]
        if test_batch.shape[1] == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[1] == 4:
            e3 = ents[:, 2]
            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[1] == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]

            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[1] == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[1] == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[1] == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[1] == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[1] == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        return score

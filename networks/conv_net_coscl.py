import sys
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, use_TG):
        super().__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla
        self.nExpert = 10
        self.nc = 16
        self.last = torch.nn.ModuleList()
        self.s_gate = 1
        self.use_TG = use_TG

        self.net1 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(1016, 2000),
            #nn.ReLU()
        )
        self.fc1 = nn.Linear(1016, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc2 = torch.nn.Embedding(len(self.taskcla), 2000)

        self.net2 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1016, 2000)
        self.fc4 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc4 = torch.nn.Embedding(len(self.taskcla), 2000)

        self.net3 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(1016, 2000)
        self.fc6 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc6 = torch.nn.Embedding(len(self.taskcla), 2000)

        self.net4 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc7 = nn.Linear(1016, 2000)
        self.fc8 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc8 = torch.nn.Embedding(len(self.taskcla), 2000)


        self.net5 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc9 = nn.Linear(1016, 2000)
        self.fc10 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc10 = torch.nn.Embedding(len(self.taskcla), 2000)


        self.net6 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc11 = nn.Linear(1016, 2000)
        self.fc12 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc12 = torch.nn.Embedding(len(self.taskcla), 2000)

        self.net7 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc13 = nn.Linear(1016, 2000)
        self.fc14 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc14 = torch.nn.Embedding(len(self.taskcla), 2000)


        self.net8 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc15 = nn.Linear(1016, 2000)
        self.fc16 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc16 = torch.nn.Embedding(len(self.taskcla), 2000)


        self.net9 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc17 = nn.Linear(1016, 2000)
        self.fc18 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc18 = torch.nn.Embedding(len(self.taskcla), 2000)


        self.net10 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nc),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*2),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*4),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.nc*8),
            nn.ReLU(),
            nn.Conv2d(self.nc*8, 254, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(254),
            nn.ReLU()
        )
        self.fc19 = nn.Linear(1016, 2000)
        self.fc20 = nn.Linear(2000, 2000)
        if self.use_TG:
            self.efc20 = torch.nn.Embedding(len(self.taskcla), 2000)


        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2000, n))
            self.ncls = n

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.relu = torch.nn.ReLU()
        self.sig_gate = torch.nn.Sigmoid()

    def forward(self, x, t, return_expert=False, avg_act=False):
        if self.use_TG: #with task adaptive gate
            masks = self.mask(t, s=self.s_gate)
            gfc1, gfc2, gfc3, gfc4, gfc5, gfc6, gfc7, gfc8, gfc9, gfc10 = masks

            self.Experts = []
            self.Experts_feature = []

            #print(x)
            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            # h1 = h1 * gfc1.expand_as(h1)
            self.Experts_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.relu(self.fc2(h1))
            h1 = self.drop2(h1)
            h1 = h1 * gfc1.expand_as(h1)
            self.Experts.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            # h2 = h2 * gfc2.expand_as(h2)
            self.Experts_feature.append(h2)
            h2 = self.relu(self.fc3(h2))
            h2 = self.relu(self.fc4(h2))
            h2 = self.drop2(h2)
            h2 = h2 * gfc2.expand_as(h2)
            self.Experts.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            # h3 = h3 * gfc3.expand_as(h3)
            self.Experts_feature.append(h3)
            h3 = self.relu(self.fc5(h3))
            h3 = self.relu(self.fc6(h3))
            h3 = self.drop2(h3)
            h3 = h3 * gfc3.expand_as(h3)
            self.Experts.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            # h4 = h4 * gfc4.expand_as(h4)
            self.Experts_feature.append(h4)
            h4 = self.relu(self.fc7(h4))
            h4 = self.relu(self.fc8(h4))
            h4 = self.drop2(h4)
            h4 = h4 * gfc4.expand_as(h4)
            self.Experts.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h5)
            h5 = self.relu(self.fc9(h5))
            h5 = self.relu(self.fc10(h5))
            h5 = self.drop2(h5)
            h5 = h5 * gfc5.expand_as(h5)
            self.Experts.append(h5.unsqueeze(0))

            h6 = self.net6(x)
            h6 = h6.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h6)
            h6 = self.relu(self.fc11(h6))
            h6 = self.relu(self.fc12(h6))
            h6 = self.drop2(h6)
            h6 = h6 * gfc6.expand_as(h6)
            self.Experts.append(h6.unsqueeze(0))

            h7 = self.net7(x)
            h7 = h7.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h7)
            h7 = self.relu(self.fc13(h7))
            h7 = self.relu(self.fc14(h7))
            h7 = self.drop2(h7)
            h7 = h7 * gfc6.expand_as(h7)
            self.Experts.append(h7.unsqueeze(0))

            h8 = self.net8(x)
            h8 = h8.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h8)
            h8 = self.relu(self.fc15(h8))
            h8 = self.relu(self.fc16(h8))
            h8 = self.drop2(h8)
            h8 = h8 * gfc6.expand_as(h8)
            self.Experts.append(h8.unsqueeze(0))

            h9 = self.net9(x)
            h9 = h9.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h9)
            h9 = self.relu(self.fc17(h9))
            h9 = self.relu(self.fc18(h9))
            h9 = self.drop2(h9)
            h9 = h9 * gfc6.expand_as(h9)
            self.Experts.append(h9.unsqueeze(0))

            h10 = self.net10(x)
            h10 = h10.view(x.shape[0], -1)
            # h5 = h5 * gfc5.expand_as(h5)
            self.Experts_feature.append(h10)
            h10 = self.relu(self.fc19(h10))
            h10 = self.relu(self.fc20(h10))
            h10 = self.drop2(h10)
            h10 = h10 * gfc6.expand_as(h10)
            self.Experts.append(h10.unsqueeze(0))

            h = torch.cat([h_result for h_result in self.Experts], 0)
            h = torch.sum(h, dim=0).squeeze(0)  # / self.nExpert

        else: #without task adaptive gate
            self.Experts = []
            self.Experts_feature = []

            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            self.Experts_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.drop2(h1)
            self.Experts.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            self.Experts_feature.append(h2)
            h2 = self.relu(self.fc2(h2))
            h2 = self.drop2(h2)
            self.Experts.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            self.Experts_feature.append(h3)
            h3 = self.relu(self.fc3(h3))
            h3 = self.drop2(h3)
            self.Experts.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            self.Experts_feature.append(h4)
            h4 = self.relu(self.fc4(h4))
            h4 = self.drop2(h4)
            self.Experts.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            self.Experts_feature.append(h5)
            h5 = self.relu(self.fc5(h5))
            h5 = self.drop2(h5)
            self.Experts.append(h5.unsqueeze(0))

            h = torch.cat([h_result for h_result in self.Experts], 0)
            h = torch.sum(h, dim=0).squeeze(0)  # / self.nExpert

        #y = []
        #for t,i in self.taskcla:
        #    y.append(self.last[t](h))

        y = self.last[t](h)

        self.grads = {}

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        if return_expert:
            self.Experts_y = []
            for i in range(self.nExpert):
                h_exp = self.Experts[i].squeeze(0)
                y_exp = self.last[t](h_exp)
                self.Experts_y.append(y_exp)

            return y, self.Experts_y, self.Experts

        else:
            return y

    def mask(self,t,s=1):
        gfc1 = self.sig_gate(s * self.efc2(t))
        gfc2 = self.sig_gate(s * self.efc4(t))
        gfc3 = self.sig_gate(s * self.efc6(t))
        gfc4 = self.sig_gate(s * self.efc8(t))
        gfc5 = self.sig_gate(s * self.efc10(t))
        gfc6 = self.sig_gate(s * self.efc12(t))
        gfc7 = self.sig_gate(s * self.efc14(t))
        gfc8 = self.sig_gate(s * self.efc16(t))
        gfc9 = self.sig_gate(s * self.efc18(t))
        gfc10 = self.sig_gate(s * self.efc20(t))

        return [gfc1,gfc2,gfc3,gfc4,gfc5,gfc6,gfc7,gfc8,gfc9,gfc10]

import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
args = get_args()

class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        KLD_loss = 0
        for k in range(len(x)):
            for l in range(len(x)):
                if l != k:
                    KLD_loss += self.criterion_KLD(F.log_softmax(x[k], dim=1), F.softmax(x[l], dim=1).detach())

        return KLD_loss

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None,log_name = None, use_TG=False):
        self.model=model
        self.model_old=model
        self.fisher=None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min * 1/3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.lamb=args.lamb
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.lamb3 = args.lamb3
        self.use_TG = use_TG
        self.model.s_gate = args.s_gate

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        
        self.omega = {}
        for n,_ in self.model.named_parameters():
            self.omega[n] = 0

        self.kld = KLD()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            
            num_batch = xtrain.size(0)
            
            self.train_epoch(t,xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            #save log for current task & old tasks at every epoch

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')

            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # Task-Adaptive Gate
        if self.use_TG:
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            mask = self.model.mask(task, s=self.model.s_gate)
            for i in range(len(mask)):
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)

        self.old_param = {}
        for n, p in self.model.named_parameters():
            self.old_param[n] = p.data.clone().detach()

        self.omega_update(t,xtrain)
        
        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward current model
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            outputs, outputs_expert, feature_expert = self.model.forward(images, task, return_expert=True)

            loss=self.criterion(t,outputs,targets) + self.lamb1 * self.kld(outputs_expert)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            output = self.model.forward(images, task)
            #output = self.model.forward(images)[t]
                
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for name, param in self.model.named_parameters():
                loss_reg+=torch.sum(self.omega[name]*(self.old_param[name] - param).pow(2))/2
        return self.ce(output,targets) + self.lamb * loss_reg
    
    def omega_update(self,t,x):
        sbatch = 20
        
        # Compute
        self.model.train()
        for i in tqdm(range(0,x.size(0),sbatch),desc='Omega',ncols=100,ascii=True):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
            images = x[b]
            # Forward and backward
            self.model.zero_grad()

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda())
            # Forward
            outputs = self.model.forward(images, task)
            #outputs = self.model.forward(images)[t]

            # Sum of L2 norm of output scores
            loss = torch.sum(outputs.norm(2, dim = -1))

            loss.backward()

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n]+= p.grad.data.abs() / x.size(0)

        return
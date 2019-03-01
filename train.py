import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import os
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=100, help='batch size')
parser.add_argument('--noise_length', '-NL', type=int, default=62)
parser.add_argument('--label_length', '-LL', type=int, default=10)
parser.add_argument('--latent_length', '-CL', type=int, default=2)
parser.add_argument('--gpu', default='1', type=str, metavar='O', help='id of gpu used')
parser.add_argument('--epochs', default=200, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size

# dataset
dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# model define
Generator = Generator()
D = D()
Q = Q()
SharedNet = sharedNet()

model = [Generator, D, Q, SharedNet]
for i in model:
  i.cuda()
  i.apply(weights_init)

# variable define
real_x = torch.FloatTensor(batch_size, 1, 28, 28).cuda()
label = torch.FloatTensor(batch_size, 1).cuda()
dis_c = torch.FloatTensor(batch_size, 10).cuda()
con_c = torch.FloatTensor(batch_size, 2).cuda()
noise = torch.FloatTensor(batch_size, 62).cuda()
real_x = Variable(real_x)
label = Variable(label, requires_grad=False)
dis_c = Variable(dis_c)
con_c = Variable(con_c)
noise = Variable(noise)

# loss
criterionD = nn.BCELoss().cuda()
criterionQ_dis = nn.CrossEntropyLoss().cuda()
criterionQ_con = NormalNLLLoss()

# optimizer
optimD = optim.Adam([{'params': SharedNet.parameters()}, {'params':D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
optimG = optim.Adam([{'params':Generator.parameters()}, {'params':Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

# fixed random variables
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)

c1 = np.hstack([c, np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c])

print('c1\n')
print(c1)
print('c2\n')
print(c2)

idx = np.arange(10).repeat(10)
one_hot = np.zeros((batch_size, 10))
one_hot[range(batch_size), idx] = 1
fix_noise = torch.Tensor(batch_size, 62).uniform_(-1, 1)

# train the network
for epoch in range(args.epochs):
    for num_iters, batch_data in enumerate(dataloader, 0):
        # Discriminator
        optimD.zero_grad()
        x, _ = batch_data

        real_x.data.resize_(x.size())
        label.data.resize_(x.size(0), 1)
        dis_c.data.resize_(x.size(0), args.label_length)
        con_c.data.resize_(x.size(0), args.latent_length)
        noise.data.resize_(x.size(0), args.noise_length)

        # Discriminator loss for real data
        real_x.data.copy_(x)
        shared_out1 = SharedNet(real_x)
        probs_real = D(shared_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # Discriminator loss for fake data
        z, idx = noise_sample(dis_c, con_c, noise, x.size(0))
        fake_x = Generator(z)   # z has been already concated
        shared_out2 = SharedNet(fake_x.detach())  # detach(): stop the BP after D
        probs_fake = D(shared_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        optimD.step()

        # G AND Q
        optimG.zero_grad()
        shared_out3 = SharedNet(fake_x)
        probs_fake = D(shared_out3)
        label.data.fill_(1.0)

        # Generator's reconstruction loss
        reconstruct_loss = criterionD(probs_fake, label)

        # latent loss
        q_logits, q_mu, q_var = Q(shared_out3)
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.1

        G_loss = reconstruct_loss + dis_loss + con_loss
        G_loss.backward()
        optimG.step()

        if num_iters % 100 == 0:
            print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                epoch, num_iters, D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy())
            )

            noise.data.copy_(fix_noise)
            dis_c.data.copy_(torch.Tensor(one_hot))

            con_c.data.copy_(torch.from_numpy(c1))
            z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
            x_save = Generator(z)
            save_image(x_save.data, './sample/c1_epoch{0}_iter{1}.png'.format(epoch, num_iters), nrow=10)

            con_c.data.copy_(torch.from_numpy(c2))
            z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
            x_save = Generator(z)
            save_image(x_save.data, './sample/c2_epoch{0}_iter{1}.png'.format(epoch, num_iters), nrow=10)



















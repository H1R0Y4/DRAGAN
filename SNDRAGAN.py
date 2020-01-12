import torch
from torch import nn

import argparse
import os
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import pylab
# from model import Generator, Discriminator,weights_init_normal
import cloudpickle



parser = argparse.ArgumentParser(description="test")
parser.add_argument("--size", default = 4,type=int, help = "64x64-> 4(default) ,128x128 -> 8 ,")
parser.add_argument("--img_path", type = str, help = " image_path")
parser.add_argument("--out_dir", type = str, help = "out_dir")
parser.add_argument("--n_epochs", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")
parser.add_argument("--batch_size", default = 100, type = int, help = "batch size")

args = parser.parse_args()
size = args.size
out_dir = args.out_dir
n_epochs = args.n_epochs
interval = args.interval
batch_size = args.batch_size


lr              = 1e-4
b1              = 0.5
b2              = 0.999
seed            = 123
C               = 10.0 #初期論文では0.5だったが、最終版v5では10に
latent_dim      = 100
lambda_gp       = 10.0 #初期論文から変更なし

img_path = glob.glob(args.img_path+"/*.png")

if not os.path.exists("./images/"):
    os.mkdir("./images/")
if not os.path.exists("./model/"):
    os.mkdir("./model/")

image_out_dir = './images/'+out_dir
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)
if not os.path.exists(f"./images/{out_dir}/loss"):
    os.mkdir(f"./images/{out_dir}/loss")
model_dir = 'model/'+out_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
discriminator_model_dir = 'model/'+out_dir+'/discriminator/'
if not os.path.exists(discriminator_model_dir):
    os.mkdir(discriminator_model_dir)
generator_model_dir = 'model/'+out_dir+'/generator/'
if not os.path.exists(generator_model_dir):
    os.mkdir(generator_model_dir)

def loss_func_dcgan_dis_real(h):
    return torch.sum(F.softplus(-h)) / batch_size

def loss_func_dcgan_dis_fake(h):
    return torch.sum(F.softplus(h)) / batch_size    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


        
class Generator(nn.Module):
    
    def __init__(self,base = 64):
        super(Generator,self).__init__()
        self.l0 = nn.Sequential(
            nn.Linear(100,size*size*base * 8,bias=False),
            nn.ReLU()
        )
#         self.deconv_block1 = nn.Sequential(
            
#             nn.ConvTranspose2d(base * 16,base * 8,4,2,1),
#             nn.ReLU(base * 8)
#         )
        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(base * 8,base * 4,4,2,1),
            nn.ReLU()
        )
        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(base * 4,base * 2,4,2,1),
            nn.ReLU()
        )
        self.deconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(base * 2,base,4,2,1),
            nn.ReLU()
        )
        self.deconv_block5 = nn.Sequential(
            nn.ConvTranspose2d(base,3,4,2,1),
            nn.Tanh()
        )
        
    def forward(self,z):
        h = self.l0(z)
        h = h.view(h.data.shape[0],512,size,size)
#         h = self.deconv_block1(h)
        h = self.deconv_block2(h)
        h = self.deconv_block3(h)
        h = self.deconv_block4(h)
        h = self.deconv_block5(h)
        
        return h


class Discriminator(nn.Module):
    
    def __init__(self,base = 64):
        super(Discriminator,self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3,base,4,2,1)),
            nn.LeakyReLU()
        )
        self.conv_block2 = nn.Sequential(   
            nn.utils.spectral_norm(nn.Conv2d(base,base * 2,4,2,1)),
            nn.LeakyReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(base * 2,base * 4,4,2,1)),
            nn.LeakyReLU()
        )
        self.conv_block4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(base * 4,base * 8,4,2,1)),
            nn.LeakyReLU()
        )
#         self.conv_block5 = nn.Sequential(
#             nn.Conv2d(base * 8,base * 16,4,2,1),
#             nn.LeakyReLU(base*16),
#         )
        self.l0 = nn.Sequential(
            nn.Linear(size*size*base*8,1),
        )
        
    def forward(self,x):
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.conv_block4(h)
#         h = self.conv_block5(h)
        h = h.view(h.shape[0],-1)
        h = self.l0(h)
        
        return h



torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)

class TrainDataset(Dataset):
    
    def __init__(self,path,transforms=None):
        super(TrainDataset,self).__init__()
        self.path = path
        self.transforms = transforms
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,idx):
        img = Image.open(self.path[idx]).convert("RGB")
#         print(idx)
        if self.transforms is not None:
            img_trans = self.transforms(img)
        return img_trans,idx

transform = transforms.Compose(
    [transforms.ToTensor()]
)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

dataset = TrainDataset(img_path,transforms=transform)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

device = 'cuda:0'
gen_model = Generator().to(device)
dis_model = Discriminator().to(device)
gen_model.apply(weights_init_normal)
dis_model.apply(weights_init_normal)
optimizer_G = torch.optim.Adam(gen_model.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(dis_model.parameters(), lr=lr, betas=(b1, b2))
criterion = nn.BCEWithLogitsLoss(reduction='mean')

gen_model.train()
dis_model.train()

check_z = torch.empty(100,100, dtype=torch.float32).uniform_(-1,1).to(device)

z = Tensor(batch_size, latent_dim).to(device)
labels = Tensor(batch_size,1).to(device)
# ----------
#  Training
# ----------
d_loss_list = []
g_loss_list = []
for epoch in range(n_epochs):
    sum_dis_loss = 0.0
    sum_gen_loss = 0.0
    for i, (imgs,idx) in enumerate(dataloader):
        if imgs.size()[0] != batch_size:
            break
        
        #本物をGPUにのせる
        x_dis = imgs.to(device)
        # -----------------
        #  Train Discriminator
        # -----------------
        for k in range(4):
            dis_model.zero_grad()

            #真の画像を判定
            y_real = dis_model(x_dis)
            
            #偽の画像を生成して判定
            z.data.normal_(0, 1)
            x_fake = gen_model(z).detach()
            y_fake = dis_model(x_fake)
            
            # 真の画像にノイズを混ぜて、gradient_penaltyを計算

            std_data = x_dis.std(dim=0, keepdims = True).to(device)

            rnd_x = torch.rand(x_dis.shape).to(device)

            x_perturbed = Tensor(x_dis + C *std_data*rnd_x).to(device)
            x_perturbed = Variable(x_perturbed, requires_grad=True)

            y_perturbed = dis_model(x_perturbed)

            fake = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

            gradients = autograd.grad(
                outputs=y_perturbed,
                inputs=x_perturbed,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            #損失を計算
            ##gradient_penaltyの損失
            grad = torch.sqrt(gradients.reshape(len(gradients), -1).norm(dim=1)**2)
            loss_gp = lambda_gp * F.mse_loss(grad,torch.ones_like(grad))

            loss_dis = loss_func_dcgan_dis_real(y_real) + loss_func_dcgan_dis_fake(y_fake) + loss_gp

            loss_dis.backward()
            optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        #偽の画像を生成して判定
        gen_model.zero_grad()
        z.data.normal_(0,1)
        x_fake = gen_model(z)       
        y_fake = dis_model(x_fake)

        loss_gen = loss_func_dcgan_dis_real(y_fake)
        loss_gen.backward()
        optimizer_G.step()
        
        #-----------
        #  記録
        #-----------

        sum_dis_loss += loss_dis.item()
        sum_gen_loss += loss_gen.item()
        d_loss_list.append(loss_dis.item())
        g_loss_list.append(loss_gen.item())

        if epoch%interval==0 and i ==0:
            with open(f'{discriminator_model_dir}dis_model_.pkl', 'wb') as f:
                cloudpickle.dump(dis_model, f)
            with open(f'{generator_model_dir}gen_model_epoch.pkl', 'wb') as f:
                cloudpickle.dump(gen_model, f)
            gen_model.eval()
            with torch.no_grad():
                generated_img = gen_model(check_z)
            save_image(generated_img,f"images/{out_dir}/{epoch:04d}.png",nrow=10)
    calc_loss_dis = sum_dis_loss/len(img_path)
    calc_loss_gen = sum_gen_loss/len(img_path)
    print(f'epoch : {epoch} dis_loss : {calc_loss_dis:.4f} gen_loss : {calc_loss_gen:.4f}')
    with open(f'./images/{out_dir}/loss/result.txt', 'a') as f:
        f.write(f"epoch             :{epoch}\ndiscriminator loss:{calc_loss_dis:.6f}\ngenerator loss    :{calc_loss_gen:.6f}\n")
# plot learning curve
plt.figure()
plt.plot(range(len(d_loss_list))[::10], d_loss_list[::10], 'r-', label='dis_loss',alpha=0.5)
plt.plot(range(len(g_loss_list))[::10], g_loss_list[::10], 'b-', label='gen_loss',alpha=0.5)
plt.legend()
plt.xlabel('iterator')
plt.ylabel('loss')
plt.grid()
plt.savefig(f"./images/{out_dir}/loss/loss.png")


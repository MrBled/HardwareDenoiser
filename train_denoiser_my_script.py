from datetime import datetime
import os
import torch
from torchvision.utils import save_image
from torch.nn import functional as F
from math import log10
import math
import time
from torch import nn
import torchvision
from datasets import Syn_NTIRE
import numpy as np
import sys
from options.train_options import TrainOptions
import path_config

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from datasets import Syn_NTIRE, test_data_NTIRE
from my_models import UnetGenerator_hardware
sys.argv = [
    'train.py',  # Placeholder script name
    '--dataroot', './datasets/maps',
    '--name', 'denoiser_pix2pix',
    '--model', 'pix2pix',
    '--gpu_ids', '3',  # Set other options here
    '--n_epochs', '100',
    '--n_epochs_decay', '100'
]

experiment_name = "Mar16_gaussian_competition_largerwienernet_64x64"
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def test_psnr(ground_truth, denoised):
    criterion = nn.MSELoss()
    avg_psnr = 0
    with torch.no_grad():
        for gt, densd in zip(ground_truth, denoised):
            mse = criterion(densd, gt)
            psnr = 10 * log10(1 / mse.item())
            torch.cuda.empty_cache()
            avg_psnr += psnr

    return avg_psnr / len(ground_truth)

def replace_tanh_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Tanh):
            print(f"Replacing Tanh at {name} with ReLU")
            setattr(module, name, nn.ReLU())
        else:
            replace_tanh_with_relu(child)




def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

def remove_batchnorm_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            print(f"Removing BatchNorm: {name}")
            setattr(model, name, nn.Identity())
        else:
            remove_batchnorm_layers(module)

def remove_dropout_layers(model):
    """
    Recursively removes all nn.Dropout layers from a model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            print(f"Removing Dropout layer: {name}")
            setattr(model, name, nn.Identity())
        else:
            remove_dropout_layers(module)

def replace_bn(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            print('replaced: ', name, attr_str)
            new_bn = torch.nn.GroupNorm(32, target_attr.num_features)
            setattr(module, attr_str, new_bn)

    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)


def load_ckpt(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


def get_std(noise_imgs, clean_img):
    num_samples = noise_imgs.shape[1]
    noise_imgs = noise_imgs.swapaxes(0,1)
    noise_only = torch.ones(noise_imgs.shape).cuda()
    for i, noise_img in enumerate(noise_imgs):
        noise_only[i] = noise_img - clean_img
    noise_square = noise_only ** 2
    noise_sum = torch.sum(noise_square, 0)
    noise_mean = noise_sum / num_samples
    noise_std = torch.sqrt(noise_mean)
    return noise_std


def get_psd(noise_imgs, clean_img, std_mean):
    num_samples = noise_imgs.shape[1]
    noise_imgs = noise_imgs.swapaxes(0, 1)
    noise_norm = torch.ones(noise_imgs.shape).cuda()
    for i, noise_img in enumerate(noise_imgs):
        noise_norm[i] = (noise_img - clean_img) / (std_mean + 0.0001)
    noise_norm = noise_norm / num_samples
    return noise_norm

# checkpoint(curr_epch, train_loss, denoiser_model,
#                        optimizer, path, text_path, scheduler, ave_psnr)
def checkpoint(epoch, train_loss, model, optimizer, path, text_path, scheduler, final_psnr):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss
            }, path + "/model_epoch_{}.pt".format(epoch))
    with open(text_path, 'a') as myfile:
        myfile.write("Epoch: {} \tLoss: {:.6f} \tPSNR: {:.2f} dB\n".format(epoch, train_loss, final_psnr))



def checkpoint_test(test_loader, model, best, epoch, best_epoch, text_path,denoiser_model, device, 
                    curr_epch, optimizer, path, train_loss, scheduler):
    psnr = 0.0
    denoiser_model.eval()
    with torch.no_grad():
        for index, (noise_img, clean_img) in enumerate(test_loader):
            noise_img = noise_img.to(device)
            clean_img = clean_img.to(device)
            denoised = denoiser_model(noise_img)
            denoised = torch.clamp(denoised, 0, 1)
            psnr += test_psnr(clean_img, denoised)

        ave_psnr = psnr / len(test_loader)
        
        if ave_psnr > best:
            best = ave_psnr
            best_epoch = i
            print(f" New best ".center(50, "*"))
            checkpoint(curr_epch, train_loss, denoiser_model,
                       optimizer, path, text_path, scheduler, ave_psnr)
            
        print(f"Curr PSNR: {ave_psnr: .2f} \t Best: {best: .2f}\t Best epoch: {best_epoch}\t Curr LR: {get_lr(optimizer): .6f}")
        return best, best_epoch


def train_gray(epoch, data_loader, device, optimizer,
               scheduler, denoiser_model):
    denoiser_model.train()

    train_loss = 0.0
    # start_time = time.perf_counter()
    for (noise_img, clean_img) in data_loader:

        noise_img = noise_img.to(device)
        clean_img = clean_img.to(device)
        denoised = denoiser_model(noise_img)
        # denoised = torch.clamp(wiener_fildenoisedered, 0, 1)
        im_loss = F.l1_loss(denoised, clean_img)

        train_loss += im_loss.item()
        optimizer.zero_grad()
        im_loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss = train_loss / len(data_loader)

    print(f"Epoch {epoch} \tTotal loss: {train_loss:.4f}")
    # checkpoint(i, loss.item(), wiener, optimizer, path, text_path)
    # checkpoint_noise_predictor(i, loss.item(), model, optimizer, path, text_path)
    return train_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def checkpoint_noise_predictor(epoch, train_loss, model, optimizer, path, text_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'noise_model_state_dict': noise_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
            }, path+"/model_epoch_{}.pt".format(epoch))
    print("Epoch saved")
    with open(text_path, 'a') as myfile:
        myfile.write("Epoch: {} \tLoss: {}\n".format(epoch, train_loss))


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('ConvBlock') != -1:
        pass
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights_ones(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_printoptions(linewidth=120)
    now = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    current_time = now.strftime("%H_%M_%S")

    dataset = Syn_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/train_imgs/', 150, patch_size=256) + Syn_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/DIV2K/', 150, patch_size=256)
    # test_dataset = test_data_NTIRE('/home/bledc/dataset/denoising_challenge/test_imgs/', 100, patch_size=256) #center crop excluded atm

    test_dataset = test_data_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/test_imgs/', 100, patch_size=256) #center crop excluded atm
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64,
        shuffle=True, num_workers=6,
        pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=3)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    print(device)

    
    model = create_model(TrainOptions().parse())
    denoiser_model = UnetGenerator_hardware(3, 3, 8).to(device)
    # denoiser_model = model.netG  # assuming the generator is the denoiser model
    # remove_batchnorm_layers(denoiser_model)
    # replace_tanh_with_relu(denoiser_model)
    # remove_dropout_layers(denoiser_model)
    learning_rate = 1e-3
    
    optimizera = torch.optim.Adam(denoiser_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizera, T_0=5000, T_mult=2)


    loss = 11
    start_time = time.perf_counter()
    best_epoch = 0
    i = 0
    best = 0
    num_epochs = 5000

    print("Commencing training")
    for i in range(0, num_epochs + 1):

        train_loss = train_gray(i, data_loader, device, optimizera,
                                scheduler, denoiser_model)
        if i % 10 == 0:
            if i ==0:
                experiment_path, current_time = path_config.get_experiment_dir()
                os.makedirs(experiment_path, exist_ok=True)
                text_path = f"{experiment_path}/{current_time}.txt"
                with open(text_path, 'w') as txt_data:
                    txt_data.write("Training Log\n")
            best, best_epoch = checkpoint_test(test_loader, model, best, i, best_epoch, text_path, denoiser_model, 
                                               device, i, optimizera, experiment_path, train_loss, scheduler)
        # scheduler.step()

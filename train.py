import argparse
import itertools
import os
import time
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from histloss import HistogramLoss # https://github.com/stat-ml/hist-loss || pip install hist-loss

import lib.pytorch_ssim as pytorch_ssim
from lib.data import get_training_set, is_image_file, get_Low_light_training_set
from lib.utils import TVLoss, print_network
from model import PLN

Name_Exp = 'PLN'
exp = Experiment(Name_Exp,save_git_info=False)
# exp.observers.append(MongoObserver(url='Host:27017', db_name='low_light'))
exp.add_source_file("train.py")
exp.add_source_file("model.py")
exp.add_source_file("lib/dataset.py")
exp.captured_out_filter = apply_backspaces_and_linefeeds


@exp.config
def cfg():
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.0001')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=48, help='Size of cropped LR image')
    parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
    parser.add_argument('--isdimColor', default=True, help='synthesis at HSV color space')
    parser.add_argument('--isaddNoise', default=True, help='synthesis with noise')
    opt = parser.parse_args()


def checkpoint(model1,model2, epoch, opt):
    try:
        os.stat(opt.save_folder)
    except:
        os.mkdir(opt.save_folder)

    model_out_path1 = opt.save_folder + "Lighten_{}_{}.pth".format(Name_Exp, epoch)
    torch.save(model1.state_dict(), model_out_path1)
    model_out_path2 = opt.save_folder + "Darken_{}_{}.pth".format(Name_Exp, epoch)
    torch.save(model2.state_dict(), model_out_path2)

    print("Checkpoint saved to {}".format(model_out_path1))
    return model_out_path1,model_out_path2


def log_metrics(_run, logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        _run.log_scalar(key, float(value), iter)
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)


def eval(model, epoch):
    print("==> Start testing")
    tStart = time.time()
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    model.eval()
    test_LL_folder = "datasets/CFFI_LOL/test/low/"
    test_NL_folder = "datasets/CFFI_LOL/test/high/"
    test_est_folder = "outputs/eopch_%04d/" % (epoch)
    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            LL = trans(Image.open(test_LL_list[i]).resize((512,512)).convert('RGB')).unsqueeze(0).cuda()
            prediction = model(LL)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            LL = LL.data[0].cpu().numpy().transpose(channel_swap)
            LL = LL * 255.0
            LL = LL.clip(0, 255)
            Image.fromarray(np.uint8(prediction+LL)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        gt = cv2.resize(gt,(512,512))
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    return psnr_score, ssim_score


@exp.automain
def main(opt, _run):
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True
    gpus_list = range(opt.gpus)

    # =============================#
    #   Prepare training data     #
    # =============================#
    # first use the synthesis data (from VOC 2007) to train the model, then use the LOL real data to fine tune
    print('===> Prepare training data')
    #train_set = get_Low_light_training_set(upscale_factor=1, patch_size=opt.patch_size, data_augmentation=True)
    train_set = get_training_set("datasets/CFFI_LOL/train", 1, opt.patch_size, True) # uncomment it to do the fine tuning
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True, drop_last=True)
    # =============================#
    #          Build model        #
    # =============================#
    print('===> Build model')

    lighten = PLN(input_dim=3, dim=64)
    darken = PLN(input_dim=3, dim=64)
    lighten = torch.nn.DataParallel(lighten)
    darken = torch.nn.DataParallel(darken)
    print("------------Load VOC preparing model--------------")
    lighten.load_state_dict(torch.load("Lighten_pretrained_best_psnr2140-ssim8001.pth", map_location=lambda storage, loc: storage), strict=True)
    darken.load_state_dict(torch.load("Darken_pretrained_best_psnr2140-ssim8001.pth", map_location=lambda storage, loc: storage), strict=True)
    print('---------- Networks architecture -------------')
    print_network(lighten)

    print('----------------------------------------------')
    if cuda:
        lighten = lighten.cuda()
        darken = darken.cuda()

    # =============================#
    #         Loss function       #
    # =============================#
    L1_criterion = nn.L1Loss()
    TV_loss = TVLoss()
    mse_loss = torch.nn.MSELoss()
    ssim = pytorch_ssim.SSIM()
    hist_loss = HistogramLoss()
    if cuda:
        gpus_list = range(opt.gpus)
        mse_loss = mse_loss.cuda()
        L1_criterion = L1_criterion.cuda()
        TV_loss = TV_loss.cuda()
        ssim = ssim.cuda(gpus_list[0])

    # =============================#
    #         Optimizer            #
    # =============================#
    parameters = [lighten.parameters()]+[darken.parameters()]
    optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    # =============================#
    #         Training             #
    # =============================#
    psnr_score, ssim_score = eval(lighten, 0)
    print(psnr_score)

    psnr_best = 0
    ssim_best = 0
    best_epoch = 0

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        print('===> training epoch %d' % epoch)
        epoch_loss = 0
        lighten.train()

        tStart_epoch = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            over_Iter = epoch * len(training_data_loader) + iteration
            optimizer.zero_grad()

            LL_t, NL_t = batch[0], batch[1]
            if cuda:
                LL_t = LL_t.cuda()
                NL_t = NL_t.cuda()

            t0 = time.time()
            #正向的照亮生产，反向的变黑生成，比较两个生成的结果
            pred_t = lighten(LL_t)
            pred_d = darken(NL_t)
            l1_loss_l = L1_criterion(LL_t+pred_t,NL_t)
            l1_loss_l += L1_criterion(torch.abs(NL_t - pred_t), LL_t)
            l1_loss_d = L1_criterion(torch.abs(NL_t - pred_d), LL_t)
            l1_loss_d += L1_criterion(LL_t+pred_d,NL_t)

            hist_loss_l = hist_loss((LL_t + pred_t),NL_t)
            hist_loss_d = hist_loss((LL_t + pred_d),NL_t)

            l1_loss = L1_criterion(pred_d,pred_t)

            ssim_loss_l = 1-ssim(LL_t + pred_t, NL_t)
            ssim_loss_l += 1-ssim(torch.abs(NL_t - pred_t), LL_t)
            ssim_loss_d = 1-ssim(torch.abs(NL_t - pred_d), LL_t)
            ssim_loss_d += 1-ssim(LL_t + pred_d, NL_t)

            ssim_loss = L1_criterion(pred_d, pred_t)

            tv_loss = TV_loss(pred_t)
            loss = 0.2*(l1_loss_l + l1_loss_d+l1_loss)+0.7*(ssim_loss_l+ssim_loss_d+ssim_loss)+0.1*(hist_loss_d+hist_loss_l)+0.001 * tv_loss #

            loss.backward()
            optimizer.step()
            t1 = time.time()

            epoch_loss += loss

            if iteration % 10 == 0:
                print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                      end=" ==> ")
                logs = {
                    "loss": loss.data,
                    "l1_loss": (l1_loss_l + l1_loss_d+l1_loss).data,
                    "ssim_loss": (ssim_loss_l+ssim_loss_d+ssim_loss).data,
                    #"hist_loss": (hist_loss_l+hist_loss_d).data,
                    "tv_loss": tv_loss.data,
                }
                log_metrics(_run, logs, over_Iter)
                print("time: {:.4f} s".format(t1 - t0))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
            training_data_loader), time.time() - tStart_epoch))
        _run.log_scalar("epoch_loss", float(epoch_loss / len(training_data_loader)), epoch)

        if epoch % (opt.snapshots) == 0:
            file_checkpoint1,file_checkpoint2 = checkpoint(lighten,darken, epoch, opt)
            exp.add_artifact(file_checkpoint1)
            exp.add_artifact(file_checkpoint2)

            psnr_score, ssim_score = eval(lighten, epoch)

            if psnr_score > psnr_best:
                psnr_best = psnr_score
                ssim_best = ssim_score
                best_epoch = epoch
                print('Update Best: ===》 ',epoch,psnr_best,ssim_best)
            logs = {
                "psnr": psnr_score,
                "ssim": ssim_score,
            }
            log_metrics(_run, logs, epoch, end_str="\n")

        if (epoch + 1) % (opt.nEpochs * 2 / 3) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    print('============================================')
    print('Best_epoch:',best_epoch)
    print('PSNR:',psnr_best)
    print('SSIM',ssim_best)
    print('============================================')


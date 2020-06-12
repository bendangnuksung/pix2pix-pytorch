from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from dataset import DatasetFromImages


class MyConfig():

    def __init__(self):
        self.dataset = 'facades'  # default dataset
        self.batch_size = 1  # 'training batch size'
        self.test_batch_size = 1  # testing batch size
        self.direction = 'b2a'  # a2b or b2a
        self.epoch_count = 1  # the starting epoch count
        self.niter = 100  # # of iter at starting learning rate
        self.niter_decay = 100  # '# of iter to linearly decay learning rate to zero'
        self.input_nc = 3 # input image channels
        self.output_nc = 3 # output image channels
        self.ngf = 64  # generator filters in first conv layer
        self.ndf = 64 # discriminator filters in first conv layer
        self.lr = 0.0002  #initial learning rate for adam
        self.lr_policy = 'lambda' # learning rate policy: lambda|step|plateau|cosine
        self.lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
        self.beta1 = 0.5  # beta1 for adam. default=0.5        
        self.threads = 4  # number of threads for data loader to use
        self.seed = 77  # random seed
        self.lamb = 10  #weight on L1 term in objective
        self.cuda = True  # use cuda?
        self.cuda_n = 0  # for multiple gpu
        self.input_shape = 256  #  input shape width=height=256

    def display(self):
        print("************************** Given config **************************")
        for key, val in self.__dict__.items():
            print(f'{key}\t:{val}')
        print("*"*60)


default_config = MyConfig()


def set_device(config):
    map_location = 'cpu'
    if not torch.cuda.is_available():
        raise Exception("No GPU found, set config.cuda=False to use CPU")    

    if config.cuda:
        device = torch.device('cuda:' + str(config.cuda_n))
    else:
        device = torch.device('cpu')
    return device


def set_seed(config):
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)


def train_from_images(A_images, B_images, A_test_images=None, B_test_images=None, is_cv2_image=False, model_ckpt_path='checkpoint/', config=default_config):
    config.display()
    device = set_device(config)
    set_seed(config)
    cudnn.benchmark = True
    train_set = DatasetFromImages(A_images, B_images, config.direction, is_cv2_image, config.input_shape)
    training_data_loader = DataLoader(dataset=train_set, num_workers=config.threads, batch_size=config.batch_size, shuffle=True)

    test_set, testing_data_loader = None, None
    if A_test_images is not None:
        test_set = DatasetFromImages(A_test_images, B_test_images, config.direction, is_cv2_image, config.input_shape)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=config.threads, batch_size=config.test_batch_size, shuffle=False)

    net_g = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_d = define_D(config.input_nc + config.output_nc, config.ndf, 'basic', gpu_id=device)

    criterionGAN = GANLoss().to(device)
    criterionL1  = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, config)
    net_d_scheduler = get_scheduler(optimizer_d, config)

    for epoch in range(config.epoch_count, config.niter + config.niter_decay + 1):
        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_a, real_b = batch[0].to(device), batch[1].to(device)

            fake_b = net_g(real_a)
            ######################
            # (1) Update D network
            ######################
            optimizer_d.zero_grad()
        
            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = net_d.forward(real_ab)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
           
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = net_d.forward(fake_ab)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()

            optimizer_g.step()

            if iteration % 25 == 0 :
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # test
        if testing_data_loader is not None:
            avg_psnr = 0
            for batch in testing_data_loader:
                input, target = batch[0].to(device), batch[1].to(device)

                prediction = net_g(input)
                mse = criterionMSE(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

        #checkpoint
        if epoch % 50 == 0:
            if not os.path.exists(model_ckpt_path):
                os.mkdir(model_ckpt_path)
            net_g_model_out_path = f"{model_ckpt_path}/netG_model_epoch_{epoch}.pth"
            net_d_model_out_path = f"{model_ckpt_path}/netD_model_epoch_{epoch}.pth"
            torch.save(net_g, net_g_model_out_path)
            torch.save(net_d, net_d_model_out_path)
            print("Checkpoint saved to -- ", config.dataset)







# if opt.cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")



# torch.manual_seed(opt.seed)
# if opt.cuda:
#     torch.cuda.manual_seed(opt.seed)


# print('===> Loading datasets')
# root_path = "dataset/"
# train_set = get_training_set(root_path + opt.dataset, opt.direction)
# test_set = get_test_set(root_path + opt.dataset, opt.direction)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

# device = torch.device("cuda:0" if opt.cuda else "cpu")

# print('===> Building models')
# net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
# net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

# criterionGAN = GANLoss().to(device)
# criterionL1 = nn.L1Loss().to(device)
# criterionMSE = nn.MSELoss().to(device)

# # setup optimizer
# optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# net_g_scheduler = get_scheduler(optimizer_g, opt)
# net_d_scheduler = get_scheduler(optimizer_d, opt)

# for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
#     # train
#     for iteration, batch in enumerate(training_data_loader, 1):
#         # forward
#         real_a, real_b = batch[0].to(device), batch[1].to(device)
#         fake_b = net_g(real_a)

#         ######################
#         # (1) Update D network
#         ######################

#         optimizer_d.zero_grad()
        
#         # train with fake
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = net_d.forward(fake_ab.detach())
#         loss_d_fake = criterionGAN(pred_fake, False)

#         # train with real
#         real_ab = torch.cat((real_a, real_b), 1)
#         pred_real = net_d.forward(real_ab)
#         loss_d_real = criterionGAN(pred_real, True)
        
#         # Combined D loss
#         loss_d = (loss_d_fake + loss_d_real) * 0.5

#         loss_d.backward()
       
#         optimizer_d.step()

#         ######################
#         # (2) Update G network
#         ######################

#         optimizer_g.zero_grad()

#         # First, G(A) should fake the discriminator
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = net_d.forward(fake_ab)
#         loss_g_gan = criterionGAN(pred_fake, True)

#         # Second, G(A) = B
#         loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
#         loss_g = loss_g_gan + loss_g_l1
        
#         loss_g.backward()

#         optimizer_g.step()

#         print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
#             epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

#     update_learning_rate(net_g_scheduler, optimizer_g)
#     update_learning_rate(net_d_scheduler, optimizer_d)

#     # test
#     avg_psnr = 0
#     for batch in testing_data_loader:
#         input, target = batch[0].to(device), batch[1].to(device)

#         prediction = net_g(input)
#         mse = criterionMSE(prediction, target)
#         psnr = 10 * log10(1 / mse.item())
#         avg_psnr += psnr
#     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

#     #checkpoint
#     if epoch % 50 == 0:
#         if not os.path.exists(model_ckpt_path):
#             os.mkdir(model_ckpt_path)
#         if not os.path.exists(os.path.join(model_ckpt_path, opt.dataset)):
#             os.mkdir(os.path.join(model_ckpt_path, opt.dataset))
#         net_g_model_out_path = f"{model_ckpt_path}/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
#         net_d_model_out_path = f"{model_ckpt_path}/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
#         torch.save(net_g, net_g_model_out_path)
#         torch.save(net_d, net_d_model_out_path)
#         print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

import os
from math import log10
import cv2
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from dataset import DatasetFromImages
import torchvision.transforms as transforms
from datetime import datetime

amp, SCALER = None, None
try:
    from torch.cuda import amp
    SCALER = amp.GradScaler()
except Exception as e:
    print("Error import AMP for torch: ", e)
    print("need torch version>=1.6")
    print("Continuing")


class MyConfig():

    def __init__(self):
        self.batch_size = 1  # 'training batch size'
        self.test_batch_size = 1  # testing batch size
        self.direction = 'b2a'  # a2b or b2a
        self.epoch_count = 1  # the starting epoch count
        self.niter = 100  # # of iter at starting learning rate
        self.niter_decay = 100  # '# of iter to linearly decay learning rate to zero'
        self.input_nc = 3  # input image channels
        self.output_nc = 3  # output image channels
        self.ngf = 64  # generator filters in first conv layer
        self.ndf = 64  # discriminator filters in first conv layer
        self.lr = 0.0002  # initial learning rate for adam
        self.lr_policy = 'lambda'  # learning rate policy: lambda|step|plateau|cosine
        self.lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
        self.beta1 = 0.5  # beta1 for adam. default=0.5
        self.threads = 4  # number of threads for data loader to use
        self.seed = 77  # random seed
        self.lamb = 10  # weight on L1 term in objective
        self.cuda = True  # use cuda?
        self.cuda_n = 0  # for multiple gpu
        self.input_shape = 256  # input shape width=height=256
        self.checkpoint_step = 1 # checkpoint after every N step
        self.checkpoint_path = 'checkpoint/' # checkpoint model
        self.test_image_path = 'result/' # prediction image save path
        self.use_mp = False  # use Mixed precision
        self.g_nblocks = 12

    def display(self):
        print("************************** Given config **************************")
        for key, val in self.__dict__.items():
            print(f'{key}\t:{val}')
        print("*" * 60)


default_config = MyConfig()


class Pix2Pix():

    def __init__(self, config=default_config):
        self.config = config
        self.net_g = None
        transform_list_rgb = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list_gray = [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))]
        self.transform_rgb = transforms.Compose(transform_list_rgb)
        self.transform_gray = transforms.Compose(transform_list_gray)
        self.save_image_counter = 0
        self.config.display()

    def set_device(self):
        if not torch.cuda.is_available():
            raise Exception("No GPU found, set config.cuda=False to use CPU")

        if self.config.cuda:
            device = torch.device('cuda:' + str(self.config.cuda_n))
        else:
            device = torch.device('cpu')
        return device

    def set_seed(self):
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

    def get_models(self, ckpt_path, load_only_gen=False):
        if self.config.cuda:
            self.device_name = 'cuda:' + str(self.config.cuda_n)
        else:
            self.device_name = 'cpu'
        print("#"*40)
        print("Loading Models")
        models = os.listdir(ckpt_path)
        if len(models) == 0:
            print("No models found in Checkpoint path:")
            return False

        gen_model = None
        dis_model = None
        gen_model_dict = {}
        dis_model_dict = {}

        for model_name in models:
            try:
                # model_name = model_name.replace('.pth', '.bin')
                path = os.path.join(ckpt_path, model_name)
                bin_path = path.replace('.pth', '.bin')
                model_path = bin_path if os.path.exists(bin_path) else path
                temp = model_name.split('_')
                epoch = temp[-1].split('.')[0]
                is_gen = model_name.startswith('netG')
                if is_gen:
                    gen_model_dict[int(epoch)] = model_path
                else:
                    dis_model_dict[int(epoch)] = model_path
            except:
                print('Not a model file: ', model_name)

        if gen_model_dict == {}:
            print("No Generator model found")
            return False

        if dis_model_dict == {}:
            print("No Discriminator model found")
            return False

        highest_epoch = sorted(gen_model_dict.keys())[-1]

        self.config.epoch_count = highest_epoch


        gen_model_path = gen_model_dict[highest_epoch]
        dis_model_path = dis_model_dict[highest_epoch]

        if gen_model_path.endswith('bin'):
            net_g_state_dict = torch.load(gen_model_path, map_location=self.device_name)
            self.net_g.load_state_dict(net_g_state_dict)
            self.net_g.to(self.device)
            if not load_only_gen:
                net_d_state_dict = torch.load(dis_model_path, map_location=self.device_name)
                self.net_d.load_state_dict(net_d_state_dict)
                self.net_d.to(self.device)
        else:
            self.net_g = torch.load(gen_model_path, map_location=self.device_name)
            self.net_g.to(self.device)
            if not load_only_gen:
                self.net_d = torch.load(dis_model_path, map_location=self.device_name)
                self.net_d.to(self.device)

        print(f"Loaded Generator:     {gen_model_path}")
        if not load_only_gen:
            print(f"Loaded Discriminator: {dis_model_path}")
        print("#"*40)
        return True

    def train_from_images(self, A_images, B_images, A_test_images=None, B_test_images=None, is_cv2_image=False,
                          load_last_model=True):
        """
        :param A_images: List of A Training images (type can be PIL object or numpy)
        :param B_images: List of B Training images (type can be PIL object or numpy)
        :param A_test_images: List of A Testing images (type can be PIL object or numpy) (optional)
        :param B_test_images: List of B Tetsting images (type can be PIL object or numpy) (optional)
        :param is_cv2_image: Set True if image is read in Numpy Cv2 format, else False if read by PIL
        :param self.model_ckpt_path: Checkpoint model path to be save
        :param config:  Hyperparameter from config
        """
        
        self.device = self.set_device()
        self.set_seed()
        self.model_ckpt_path = self.config.checkpoint_path
        cudnn.benchmark = True
        train_set = DatasetFromImages(A_images, B_images, self.config.direction, is_cv2_image, self.config.input_shape,
                                      self.config.input_nc, self.config.output_nc)
        training_data_loader = DataLoader(dataset=train_set, num_workers=self.config.threads, batch_size=self.config.batch_size,
                                          shuffle=True)

        test_set, testing_data_loader = None, None
        test_image_exist = False if A_test_images is None else True
        if A_test_images is not None:
            test_set = DatasetFromImages(A_test_images, B_test_images, self.config.direction, is_cv2_image, self.config.input_shape)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=self.config.threads,
                                             batch_size=self.config.test_batch_size, shuffle=False)

        self.net_g = define_G(self.config.input_nc, self.config.output_nc, self.config.ngf, 'batch', False, 'normal', 0.02, gpu_id=self.device, n_blocks=self.config.g_nblocks)
        self.net_d = define_D(self.config.input_nc + self.config.output_nc, self.config.ndf, 'basic', gpu_id=self.device)
        if load_last_model:
            self.get_models(self.model_ckpt_path)

        criterionGAN = GANLoss().to(self.device)
        criterionL1 = nn.L1Loss().to(self.device)
        criterionMSE = nn.MSELoss().to(self.device)

        optimizer_g = optim.Adam(self.net_g.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        net_g_scheduler = get_scheduler(optimizer_g, self.config)
        net_d_scheduler = get_scheduler(optimizer_d, self.config)
        total_epoch =  self.config.niter + self.config.niter_decay + 1
        start = datetime.now()
        for epoch in range(self.config.epoch_count, total_epoch):
            # train
            for iteration, batch in enumerate(training_data_loader, 1):
                # forward
                real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)

                # Train with Mixed precision
                if self.config.use_mp:
                    with amp.autocast():
                        fake_b = self.net_g(real_a)
                    ######################
                    # (1) Update D network
                    ######################
                    optimizer_d.zero_grad()

                    # train with fake
                    fake_ab = torch.cat((real_a, fake_b), 1)
                    with amp.autocast():
                        pred_fake = self.net_d.forward(fake_ab.detach())
                        loss_d_fake = criterionGAN(pred_fake, False)

                    # train with real
                    real_ab = torch.cat((real_a, real_b), 1)
                    with amp.autocast():
                        pred_real = self.net_d.forward(real_ab)
                        loss_d_real = criterionGAN(pred_real, True)

                        # Combined D loss
                        loss_d = (loss_d_fake + loss_d_real) * 0.5

                    SCALER.scale(loss_d).backward()
                    SCALER.step(optimizer_d)
                    SCALER.update()

                    ######################
                    # (2) Update G network
                    ######################

                    optimizer_g.zero_grad()

                    # First, G(A) should fake the discriminator
                    fake_ab = torch.cat((real_a, fake_b), 1)
                    with amp.autocast():
                        pred_fake = self.net_d.forward(fake_ab)
                        loss_g_gan = criterionGAN(pred_fake, True)
                        # Second, G(A) = B
                        loss_g_l1 = criterionL1(fake_b, real_b) * self.config.lamb
                        loss_g = loss_g_gan + loss_g_l1

                    SCALER.scale(loss_g).backward()
                    SCALER.step(optimizer_g)
                    SCALER.update()

                else:
                    fake_b = self.net_g(real_a)
                    ######################
                    # (1) Update D network
                    ######################
                    optimizer_d.zero_grad()
                    # train with fake
                    fake_ab = torch.cat((real_a, fake_b), 1)
                    pred_fake = self.net_d.forward(fake_ab.detach())
                    loss_d_fake = criterionGAN(pred_fake, False)
                    # train with real
                    real_ab = torch.cat((real_a, real_b), 1)
                    pred_real = self.net_d.forward(real_ab)
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
                    pred_fake = self.net_d.forward(fake_ab)
                    loss_g_gan = criterionGAN(pred_fake, True)
                    # Second, G(A) = B
                    loss_g_l1 = criterionL1(fake_b, real_b) * self.config.lamb
                    loss_g = loss_g_gan + loss_g_l1
                    loss_g.backward()
                    optimizer_g.step()

                if iteration % 25 == 0:
                    time_taken = (datetime.now() - start).total_seconds()
                    print("===> Epoch[{}/{}] Steps:({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Time: {}".format(
                        epoch, total_epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), time_taken))
                    start = datetime.now()

            update_learning_rate(net_g_scheduler, optimizer_g)
            update_learning_rate(net_d_scheduler, optimizer_d)

            # test
            if testing_data_loader is not None:
                avg_psnr = 0
                for batch in testing_data_loader:
                    input, target = batch[0].to(self.device), batch[1].to(self.device)

                    prediction = self.net_g(input)
                    mse = criterionMSE(prediction, target)
                    psnr = 10 * log10(1 / mse.item())
                    avg_psnr += psnr
                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

            # checkpoint
            if epoch % self.config.checkpoint_step == 0:
                if not os.path.exists(self.model_ckpt_path):
                    os.mkdir(self.model_ckpt_path)
                self.net_g.eval()
                self.net_d.eval()
                net_g_model_out_path = f"{self.model_ckpt_path}/netG_model_epoch_{epoch}.pth"
                net_d_model_out_path = f"{self.model_ckpt_path}/netD_model_epoch_{epoch}.pth"
                # torch.save(self.net_g.state_dict(), net_g_model_out_path.replace('.pth', '.bin'))
                # torch.save(self.net_d.state_dict(), net_d_model_out_path.replace('.pth', '.bin'))
                torch.save(self.net_g, net_g_model_out_path)
                torch.save(self.net_d, net_d_model_out_path)
                print("Checkpoint saved to -- ", self.model_ckpt_path)

                # prediction
                if self.config.direction == 'b2a':
                    if test_image_exist:
                        image = random.choice(B_test_images)
                    else:
                        image = random.choice(B_images)

                else:
                    if test_image_exist:
                        image = random.choice(A_test_images)
                    else:
                        image = random.choice(A_images)

                self.predict([image], is_cv2_image=True)
                self.net_g.train()
                self.net_d.train()
        
        print("Completed")

    def process_image(self, images, is_cv2_image):
        final_images = []
                  
        for image in images:
            if is_cv2_image:                
                if self.config.input_nc == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image).convert('RGB')
                else:
                    image = Image.fromarray(image).convert('L')
            image.resize((self.config.input_shape, self.config.input_shape), Image.BICUBIC)

            if self.config.input_nc == 3:
                image = self.transform_rgb(image)
            else:
                image = self.transform_gray(image)

            final_images.append(image)

        return final_images

    def post_process_image(self, image_tensor):
        image_numpy = image_tensor.float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image = np.array(image_pil)
        return image

    def predict(self, images, save_image_dir_path=None, model_ckpt_path=None, is_cv2_image=False, save_image=True):
        if save_image_dir_path is None:
            save_image_dir_path = self.config.test_image_path
        if model_ckpt_path is None:
            model_ckpt_path = self.config.checkpoint_path

        self.device = self.set_device()
        self.set_seed()
        if self.net_g is None:
            self.get_models(model_ckpt_path, load_only_gen=True)
        self.net_g.eval()

        images = self.process_image(images, is_cv2_image)
        output_images = []
        for image in images:
            input_image = image.unsqueeze(0).to(self.device)
            out = self.net_g(input_image)
            out_img = out.detach().squeeze(0).cpu()
            self.save_image_counter += 1
            file_path = os.path.join(save_image_dir_path, 'predict_image_' + str(self.save_image_counter) +'.jpg')
            processed_image = self.post_process_image(out_img)
            output_images.append(processed_image)
            if save_image:
                cv2.imwrite(file_path, processed_image)
                print("Image saved as {}".format(file_path))
        return output_images

    def __del__(self):  
        print("Destructor called, object deleted.")


if __name__ == '__main__':
    from PIL import Image
    import cv2

    train_a = 'dataset/facades/train/a'
    train_b = 'dataset/facades/train/b'
    filenames = os.listdir(train_a)

    train_a_images = []
    train_b_images = []

    for file in filenames:
        a = os.path.join(train_a, file)
        b = os.path.join(train_b, file)
        # a_im = Image.open(a).convert('RGB')
        # b_im = Image.open(b).convert('RGB')
        a_im = cv2.imread(a)
        b_im = cv2.imread(b, 0)
        train_a_images.append(a_im)
        train_b_images.append(b_im)

    config = MyConfig()
    config.input_shape = 512
    config.cuda_n = 0
    config.batch_size = 1
    config.input_nc = 1
    config.checkpoint_path = 'myckpt/'
    config.test_image_path = 'myckpt/'
    config.use_mp = True
    print(train_b_images[0].shape)

    pix2pix = Pix2Pix(config)

    pix2pix.train_from_images(train_a_images, train_b_images, is_cv2_image=True)
    # pix2pix.predict(train_b_images, is_cv2_image=True)

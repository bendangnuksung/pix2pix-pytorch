from os import listdir
from os.path import join
import random
import cv2

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, input_shape=256):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.input_shape = input_shape
        self.input_shape_pad = int(self.input_shape * 1.1171875) # if input shape = 256 then input_shape_pad = 256 + 30

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((self.input_shape_pad, self.input_shape_pad), Image.BICUBIC)
        b = b.resize((self.input_shape_pad, self.input_shape_pad), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, self.input_shape_pad - self.input_shape - 1))
        h_offset = random.randint(0, max(0, self.input_shape_pad - self.input_shape - 1))
    
        a = a[:, h_offset:h_offset + self.input_shape, w_offset:w_offset + self.input_shape]
        b = b[:, h_offset:h_offset + self.input_shape, w_offset:w_offset + self.input_shape]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromImages(data.Dataset):
    def __init__(self, A_images, B_images, direction, is_cv2_image=False, input_shape=256, input_nc=3, output_nc=3):
        super(DatasetFromImages, self).__init__()
        self.direction = direction
        if direction == 'b2a':
            self.a_nc = output_nc
            self.b_nc = input_nc
        else:
            self.a_nc = input_nc
            self.b_nc = output_nc

        self.A_images = A_images if not is_cv2_image else self.cvt_to_pil(A_images, self.a_nc)
        self.B_images = B_images if not is_cv2_image else self.cvt_to_pil(B_images, self.b_nc)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.input_shape = input_shape
        self.input_shape_pad = int(self.input_shape * 1.1171875) # if input shape = 256 then input_shape_pad = 256 + 30

    def cvt_to_pil(self, images, nc):
        final_images = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if nc == 3:
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image).convert('L')
            final_images.append(image)
        return final_images

    def __getitem__(self, index):
        a = self.A_images[index]
        b = self.B_images[index]
        a = a.resize((self.input_shape_pad, self.input_shape_pad), Image.BICUBIC)
        b = b.resize((self.input_shape_pad, self.input_shape_pad), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, self.input_shape_pad - self.input_shape - 1))
        h_offset = random.randint(0, max(0, self.input_shape_pad - self.input_shape - 1))
    
        a = a[:, h_offset:h_offset + self.input_shape, w_offset:w_offset + self.input_shape]
        b = b[:, h_offset:h_offset + self.input_shape, w_offset:w_offset + self.input_shape]

        if self.a_nc == 3:
            a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        else:
            a = transforms.Normalize((0.5, ), (0.5, ))(a)

        if self.b_nc == 3:
            b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        else:
            b = transforms.Normalize((0.5, ), (0.5, ))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.A_images)

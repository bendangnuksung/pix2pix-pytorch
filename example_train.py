import os 
import cv2
from PIL import Image
from train import train_from_images, MyConfig

train_a = 'dataset/facades/train/a'
train_b = 'dataset/facades/train/b'
filenames = os.listdir(train_a)


train_a_images = []
train_b_images = []

for file in filenames:
	a = os.path.join(train_a, file)
	b = os.path.join(train_b, file)
	a_im = Image.open(a)
	b_im = Image.open(b)
	train_a_images.append(a_im)
	train_b_images.append(b_im)

config = MyConfig()
config.input_shape = 512
config.cuda_n = 1
config.batch_size = 2

train_from_images(train_a_images, train_b_images, model_ckpt_path='myckpt', config=config)
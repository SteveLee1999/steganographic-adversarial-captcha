import torch
from PIL import Image
import numpy as np
from homework.gan import Generator
import torchvision.transforms as transforms
G = Generator()
G.load_state_dict(torch.load('/home/zhongyu_bishe/code/homework/state_dict/generator.pth', map_location='cpu'))
for i in range(500):
    img = Image.open('images/'+str(i)+'.jpg')
    transform = transforms.Compose(
        [transforms.ToTensor()])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    pert_img = img + G(img)
    transforms \
        .ToPILImage()(pert_img[0]) \
        .save('pert/%s.jpg' % str(i))

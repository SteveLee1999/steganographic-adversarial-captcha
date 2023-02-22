from mycaptcha.image import CHARACTERS, LEN, ImageCaptcha, WIDTH, HEIGHT, FONTS
from gan.models import Generator, CaptchaSolver
from gan.train_ocr import decode
from steganography.juniward import getCost
from steganography.stc import STC
from steganography.steganalysis.xu import CNN, dct_preprocess
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import string
import numpy as np
import random
import imageio
import jpegio
BOUND = 0.3  # 扰动范围
BOX_MIN = 0
BOX_MAX = 1
ALPHA = 0.5  # payload
BETA = 0  # 对抗组比例，初始化为0
MAX_ITER = 100
EPS = 0.5
USE_CUDA = False
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
IMG_GENERATOR = ImageCaptcha(width=WIDTH, height=HEIGHT, font=FONTS)
PERT_GENERATOR = Generator().to(DEVICE)
PERT_GENERATOR.load_state_dict(torch.load('module/GAN_R_00.pth', map_location='cpu'))
OCR = CaptchaSolver('V').to(DEVICE)
OCR.load_state_dict(torch.load('module/OCR_V.pth', map_location='cpu'))
ANALYST = CNN().to(DEVICE)
ANALYST.load_state_dict(torch.load('module/STEG_XU.pth', map_location='cpu'))


if __name__ == "__main__":

    #####################
    # module 1: captcha #
    #####################
    label = ''.join(random.sample(CHARACTERS[1:], LEN))
    img = IMG_GENERATOR.generate_image(label)
    print('TEXT: ' + label)
    img.save('captcha.png')

    #########################
    # module 2: adversarial #
    #########################
    img = torch.unsqueeze(to_tensor(img), 0).to(DEVICE)
    PERT_GENERATOR.eval()
    OCR.eval()
    for i in range(MAX_ITER):
        pert = torch.clamp(PERT_GENERATOR(img), -BOUND, BOUND)
        img = torch.clamp(img+pert, BOX_MIN, BOX_MAX)
        pred_label = decode(OCR(img).permute(1, 0, 2).argmax(dim=-1)[0])
        if pred_label != label:
            print('ATTACK OCR: '+label+'->'+pred_label)
            transforms.ToPILImage()(pert[0].detach().cpu()).save('pert.png')
            transforms.ToPILImage()(img[0].detach().cpu()).save('adversarial.png')
            break
        else:
            print('ATTACK FAIL: %d' % i)

    ###########################
    # module 3: steganography #
    ###########################
    img = transforms.ToPILImage()(img[0].detach().cpu()).convert('L')
    img.save('adversarial.jpg', 'JPEG', quality=95)
    img = np.array(img)
    spatial = imageio.imread('adversarial.jpg')
    jpg = jpegio.read('adversarial.jpg')
    rho_p, rho_m, _, _, = getCost(spatial, jpg)
    for i in range(MAX_ITER):
        # step1:像素分组
        steg_img = img
        common_pos = []
        for j in range(steg_img.shape[0]):
            for k in range(steg_img.shape[1]):
                common_pos.append((j, k))
        adjustable_pos = random.sample(common_pos, int(len(common_pos) * BETA))
        for j in adjustable_pos:
            common_pos.remove(j)

        # step2:隐写组:信息嵌入
        message = ''.join([random.choice(string.printable) for _ in range(random.randint(5, 10))]).encode('utf8')
        print('MESSAGE: '+str(message))
        common_pixel = []
        common_cost = []
        for pos in common_pos:
            common_pixel.append(steg_img[pos[0], pos[1]])
            common_cost.append(rho_p[pos[0], pos[1]])
        stc = STC()
        common_pixel = stc.embed(common_pixel, common_cost, message)
        for i in range(len(common_pos)):
            steg_img[common_pos[i][0], common_pos[i][1]] = common_pixel[i]
        transforms.ToPILImage()(steg_img[0]).save('stegnography.png')

        # step3:对抗组:攻击分类器
        steg_img = torch.from_numpy(steg_img).float().to(DEVICE)
        steg_img = torch.unsqueeze(steg_img, 0)
        steg_img = torch.unsqueeze(steg_img, 0)
        ANALYST.eval()
        EPS /= MAX_ITER
        ANALYST(dct_preprocess(steg_img, jpg, DEVICE))
        img_map = ANALYST.get_map()
        adv_img = steg_img
        for j in range(MAX_ITER):
            # term1: loss
            output = F.softmax(ANALYST(dct_preprocess(adv_img, jpg, DEVICE)), dim=1)
            label = torch.tensor([1]).to(DEVICE)
            term_1 = F.cross_entropy(output, label)
            # term2: feature map
            adv_img_map = ANALYST.get_map()
            term_2 = torch.tensor(0, dtype=torch.float32).to(DEVICE)
            for l in range(len(img_map)):
                term_2 += torch.mean(torch.norm((img_map[l] - adv_img_map[l]).flatten()))
            # term3:
            term_3 = F.conv2d(adv_img - steg_img, torch.ones((1, 1, 3, 3)).to(DEVICE))
            term_3 = torch.sum(torch.abs(term_3))
            # 汇总
            ALPHA = 1
            BETA = 1
            loss = -term_1 - ALPHA*term_2 - BETA*term_3
            ANALYST.zero_grad()
            loss.backward()
            for pos in adjustable_pos:
                adv_img[0][pos] += torch.clamp(EPS * adv_img.grad.data.sign()[0][pos], 0, 1)
            if F.softmax(ANALYST(dct_preprocess(adv_img, jpg, DEVICE)), dim=1).argmax(dim=1).item() == 0:
                print('ATTACK ANALYST!')
                exit(0)
            else:
                print('FAIL FOR ITER: %d' % j)







    '''
    img = imageio.imread(COVER_PATH)
    stego_img = imageio.imread(STEGO_PATH)
    print(cal_psnr(img, stego_img))
    f = open("output.txt", "w")
    f.write(str(stc.extract(common_pixel)))
    print(len(str(stc.extract(common_pixel))))
    '''



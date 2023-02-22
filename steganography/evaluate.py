import numpy as np
import imageio


def cal_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(255.0 / np.sqrt(mse))


if __name__ == '__main__':
    for i in range(5):
        a = imageio.imread('../steganalysis/train/0_' + str(i) + '.jpg')
        b = imageio.imread('../steganalysis/train/1_' + str(i) + '.jpg')
        print(cal_psnr(a, b))

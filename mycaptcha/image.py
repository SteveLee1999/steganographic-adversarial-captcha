import os
import torch
import numpy as np
import jpegio
import imageio
import string
import random
from captcha.image import _Captcha
from steganography.juniward import getCost
from steganography.stc import STC
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from steganography.steganalysis.xu import dct_preprocess
CHARACTERS = '-' + string.digits + string.ascii_letters
CLASS_NUM = len(CHARACTERS)
WIDTH = 192
HEIGHT = 64
LEN = 4  # 一张验证码的字符数
FONT_PATH = '/home/zhongyu_bishe/code/mycaptcha/font/'
FONTS = [FONT_PATH+'African.ttf', FONT_PATH+'Cartoon.ttf', FONT_PATH+'JoeJack.ttf', FONT_PATH+'OrangeJuice.ttf']
TABLE = []
for i in range(256):
    TABLE.append(i*1.97)


class ImageCaptcha(_Captcha):
    """
    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width, height, font=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = font
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color, number=3):
        while number:
            w, h = image.size
            x1 = random.randint(0, int(w / 3))
            x2 = random.randint(w - int(w / 3), w)
            y1 = random.randint(int(h / 3), h - int(h / 3))
            y2 = random.randint(y1, h - int(h / 3))
            points = [x1, y1, x2, y2]
            end = random.randint(160, 200)
            start = random.randint(0, 20)
            Draw(image).arc(points, start, end, fill=color)
            number -= 1
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=100):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            if random.random() > 0.5:
                images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(TABLE)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = self.random_color(238, 255)
        color = self.random_color(100, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, color, background)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im

    def random_color(self, start, end, opacity=None):
        red = random.randint(start, end)
        green = random.randint(start, end)
        blue = random.randint(start, end)
        if opacity is None:
            return (red, green, blue)
        return (red, green, blue, opacity)


class CaptchaDataset(Dataset):
    def __init__(self, length):
        super(CaptchaDataset, self).__init__()
        self.length = length

    # 魔法方法1:如果一个类表现得像一个list，可以通过__len__定义元素的数目
    def __len__(self):
        return self.length

    # 魔法方法2:index这里没有用上
    def __getitem__(self, index):
        str = ''.join(random.sample(CHARACTERS[1:], LEN))
        img = ImageCaptcha(width=WIDTH, height=HEIGHT, font=FONTS).generate_image(str)
        img = to_tensor(img)
        # eg. random_str = '05aB',
        # 有: target = [1, 6, 11, 12]表示每个字符在characters中的位置
        label = torch.tensor([CHARACTERS.find(x) for x in str], dtype=torch.long)
        return img, label


def generate_and_save(path, count):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        image = ImageCaptcha(width=WIDTH, height=HEIGHT, font=FONTS)
        random_str = ''.join(random.sample(CHARACTERS, LEN))
        Image.open(image.generate(random_str)).save(path+str(i)+'.jpg', 'JPEG', quality=95)
        print(str(i)+'_done!')


class SteganoDataset(Dataset):
    def __init__(self, length):
        super(SteganoDataset, self).__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        text = ''.join(random.sample(CHARACTERS[1:], LEN))
        img = ImageCaptcha(width=WIDTH, height=HEIGHT, font=FONTS).generate_image(text).convert('L')
        img.save(str(index)+'.jpg', 'JPEG', quality=95)
        img = torch.unsqueeze(to_tensor(img), 0).numpy()
        jpg = jpegio.read(str(index) + '.jpg')
        # img.shape = [1, 1, H, W]
        label = random.randint(0, 1)
        if label == 1:
            rho_p, rho_m, _, _, = getCost(str(index)+'.jpg')
            stc = STC()
            m = ''.join([random.choice(string.printable) for _ in range(random.randint(10, 600))]).encode('utf8')
            img_shape = img.shape
            img = stc.embed(img.flatten(), rho_p.flatten(), m).reshape(img_shape)
        ret = dct_preprocess(img, jpg)[0]
        os.remove(str(index)+'.jpg')
        return ret, label

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from mycaptcha.image import HEIGHT, WIDTH, CLASS_NUM
BATCH = 128
CHANNEL = 3


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Inception3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = models.inception.BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = models.inception.BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = models.inception.BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = models.inception.BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = models.inception.BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = models.inception.InceptionA(192, pool_features=32)
        self.Mixed_5c = models.inception.InceptionA(256, pool_features=64)
        self.Mixed_5d = models.inception.InceptionA(288, pool_features=64)
        self.Mixed_6a = models.inception.InceptionB(288)
        self.Mixed_6b = models.inception.InceptionC(768, channels_7x7=128)
        self.Mixed_6c = models.inception.InceptionC(768, channels_7x7=160)
        self.Mixed_6d = models.inception.InceptionC(768, channels_7x7=160)
        self.Mixed_6e = models.inception.InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = models.inception.InceptionAux(768, num_classes)
        self.Mixed_7a = models.inception.InceptionD(768)
        self.Mixed_7b = models.inception.InceptionE(1280)
        self.Mixed_7c = models.inception.InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=2)
        #if self.training and self.aux_logits:
        #    return x, aux
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = models.densenet._DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = models.densenet._Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 修改1：kernel_size设置为2
        out = F.avg_pool2d(out, kernel_size=2, stride=1)
        # 修改2：删除全连接层
        return out


def densenet201(pretrained=False, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        pattern = models.densenet.re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = models.densenet.model_zoo.load_url(models.densenet.model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


class CaptchaSolver(nn.Module):
    def __init__(self, model):
        super(CaptchaSolver, self).__init__()
        self.input_shape = (3, HEIGHT, WIDTH)
        self.cnn = self.choose(model)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=CLASS_NUM)

    def choose(self, model):
        if model == 'R':
            cnn = ResNet(models.resnet.Bottleneck, [3, 8, 36, 3])
            cnn.load_state_dict(torch.load("/home/zhongyu_bishe/code/gan/state_dict/PRE_R.pth", map_location='cpu'))
        elif model == 'V':
            cnn = VGG(models.vgg.make_layers(models.vgg.cfg['E'], batch_norm=True))
            cnn.load_state_dict(torch.load("/home/zhongyu_bishe/code/gan/state_dict/PRE_V.pth", map_location='cpu'))
        elif model == 'I':
            cnn = Inception3(transform_input=False)
            cnn.load_state_dict(torch.load("/home/zhongyu_bishe/code/gan/state_dict/PRE_I.pth", map_location='cpu'))
        elif model == 'D':
            cnn = densenet201(pretrained=True)
        else:
            print('ERROR!')
            return -1
        return cnn

    def infer_features(self):
        # x.shape = [B, C, H, W] (1相当于占位模拟batch_size)
        x = torch.zeros((1,) + self.input_shape)
        # x.shape -> [B, 1920, 1, 5]依旧可以对应B,C,H,W只不过是高度提取后的特征
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    # x.shape = [B, C, H, W] (B = Batch size)
    def forward(self, x):
        # x.shape = [B, 256, a, b] (a,b和之前相等)
        x = self.cnn(x)
        # x.shape = [b, B, 256*a]
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(2, 0, 1)
        # x.shape = [b, B, 256]
        x, _ = self.lstm(x)
        # x.shape = [b, B, CLASS_NUM]
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_shape = (BATCH, 3, HEIGHT, WIDTH)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d((1, 9))
        )
        self.fc = nn.Linear(self.infer_features(), 2)

    def infer_features(self):
        x = torch.zeros(self.input_shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (BATCH, -1))
        return x.shape[1]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (BATCH, -1))
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.bottle_neck = nn.Sequential(
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.2),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.layer(x)
        return x

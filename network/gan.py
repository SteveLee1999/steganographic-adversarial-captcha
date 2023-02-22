from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from homework.googlenet import GoogLeNet
BOUND = 0.3  # 扰动范围
BOX_MIN = 0
BOX_MAX = 1
BATCH = 200
EPOCHS = 2000
LEARNING_RATE = 1e-4
TRAIN_SIZE = 100
USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")


def train_batch(images, labels, optimizer_d, optimizer_g, freeze_d):
    perturbation = G(images)
    perturbation = torch.clamp(perturbation, -BOUND, BOUND)
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, BOX_MIN, BOX_MAX)

    ##########################
    # Discriminator.backward #
    ##########################
    if freeze_d:
        loss_d = torch.tensor(0)
        acc_d = 0
    else:
        optimizer_d.zero_grad()
        pred_real = F.softmax(D(images), dim=1)
        label_real_hard = torch.zeros(BATCH, dtype=torch.long).to(DEVICE)
        check_real = torch.tensor(([i == j for i, j in zip(pred_real.argmax(dim=1), label_real_hard)]), dtype=torch.float64)
        acc_d = torch.mean(check_real).item()
        label_real_soft = torch.rand(BATCH, dtype=torch.float32).to(DEVICE) / 3
        loss_real = F.binary_cross_entropy(pred_real[:,1], label_real_soft)

        pred_fake = F.softmax(D(adv_images.detach()), dim=1)
        label_fake_hard = torch.ones(BATCH, dtype=torch.long).to(DEVICE)
        check_fake = torch.tensor(([i == j for i, j in zip(pred_fake.argmax(dim=1), label_fake_hard)]), dtype=torch.float64)
        acc_d += torch.mean(check_fake).item()
        acc_d /= 2
        label_fake_soft = torch.rand(BATCH, dtype=torch.float32).to(DEVICE) / 3
        loss_fake = F.binary_cross_entropy(pred_fake[:,0], label_fake_soft)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

    ######################
    # Generator.backward #
    ######################
    optimizer_g.zero_grad()

    # loss_gan: loss of generator(to produce better perturbation)
    pred_fake = F.softmax(D(adv_images), dim=1)
    label_fake_hard = torch.zeros(BATCH, dtype=torch.long).to(DEVICE)
    check_fake = torch.tensor(([i == j for i, j in zip(pred_fake.argmax(dim=1), label_fake_hard)]), dtype=torch.float64)
    acc_g = torch.mean(check_fake).item()
    label_fake_soft = torch.rand(BATCH, dtype=torch.float32).to(DEVICE) / 10
    loss_gan = F.binary_cross_entropy(pred_fake[:,1], label_fake_soft)

    # loss_adv: adversarial attack
    outputs = T(adv_images)
    loss_adv = -nn.CrossEntropyLoss()(outputs, labels)
    _, predicted = torch.max(outputs, 1)
    acc_t = (predicted == labels).sum().item() / labels.size(0)

    # loss_hinge: bound the magnitude of the perturbation
    C = 0.1
    loss_hinge = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
    loss_hinge = torch.max(loss_hinge - C, torch.zeros(1).to(DEVICE))

    # 损失汇总：加权平均
    ALPHA = 30
    BETA = 40
    loss_g = loss_gan + loss_adv*ALPHA + loss_hinge*BETA
    loss_g.backward()
    optimizer_g.step()
    return adv_images, perturbation, loss_d.item(), loss_g.item(), acc_d, acc_g, acc_t


def train(optimizer_d, optimizer_g, epoch, dataloader):
    loss_g_mean = 0
    loss_d_mean = 0
    acc_g_mean = 0
    acc_d_mean = 0
    acc_t_mean = 0
    i = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if batch_index % 10 == 0:
                adv_images, perturbation, loss_d, loss_g, acc_d, acc_g, acc_t = \
                    train_batch(images, labels, optimizer_d, optimizer_g, False)
            else:
                adv_images, perturbation, _, loss_g, _, acc_g, acc_t = \
                    train_batch(images, labels, optimizer_d, optimizer_g, True)
                loss_d = loss_d_mean
                acc_d = acc_d_mean
            loss_d_mean = (batch_index*loss_d_mean + loss_d)/(batch_index+1)
            loss_g_mean = (batch_index * loss_g_mean + loss_g) / (batch_index + 1)
            acc_d_mean = (batch_index * acc_d_mean + acc_d) / (batch_index + 1)
            acc_g_mean = (batch_index * acc_g_mean + acc_g) / (batch_index + 1)
            acc_t_mean = (batch_index * acc_t_mean + acc_t) / (batch_index + 1)
            pbar.set_description(f'Epoch: {epoch} '
                                 #f'D_Loss: {loss_d_mean:.4f} '
                                 #f'G_Loss: {loss_g_mean:.4f} '
                                 f'D_Acc: {acc_d_mean:.4f} '
                                 f'G_Acc: {acc_g_mean:.4f} '
                                 f'T_Acc: {acc_t_mean:.4f} ')
            if epoch % 10 == 0 and batch_index < 3:
                print('save for %d' % epoch)
                i += 1
                transforms \
                    .ToPILImage()(images[0].detach().cpu()) \
                    .save('%s_org.png' % str(i))
                transforms \
                    .ToPILImage()(adv_images[0].detach().cpu()) \
                    .save('%s_per.png' % str(i))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_shape = (BATCH, 3, 32, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
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


if __name__ == '__main__':
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    T = GoogLeNet(in_channels=3, num_classes=10).to(DEVICE)
    T.load_state_dict(torch.load('/home/zhongyu_bishe/code/homework/state_dict/googlenet.pth', map_location='cpu'))
    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)
    optimizer_d = torch.optim.Adam(D.parameters(), LEARNING_RATE, amsgrad=True)
    optimizer_g = torch.optim.Adam(G.parameters(), LEARNING_RATE, amsgrad=True)
    for epoch in range(1, EPOCHS + 1):
        train(optimizer_d, optimizer_g, epoch, train_loader)
        if epoch > 100 and epoch % 10 == 0:
            torch.save(G.state_dict(), '/home/zhongyu_bishe/code/homework/state_dict/generator.pth')

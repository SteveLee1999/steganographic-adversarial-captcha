from gan.models import Generator, Discriminator, CaptchaSolver
from gan.train_ocr import decode, calc_acc
from mycaptcha.image import CaptchaDataset, CHARACTERS, CLASS_NUM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import imageio
BOUND = 0.3  # 扰动范围
BOX_MIN = 0
BOX_MAX = 1
BATCH = 128
EPOCHS = 2000
LEARNING_RATE = 1e-4
TRAIN_SIZE = 100
USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
T = CaptchaSolver('I').to(DEVICE)
T.load_state_dict(torch.load('/home/zhongyu_bishe/code/pipeline/module/OCR_I.pth', map_location='cpu'))


def get_fake_matrix(train):
    if train:
        fake_matrix = np.zeros((CLASS_NUM, CLASS_NUM))
        dataset = CaptchaDataset(100 * BATCH)
        dataloader = DataLoader(dataset, batch_size=BATCH, num_workers=12)
        for batch_index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = T(images).permute(1, 0, 2).argsort(dim=-1).detach().cpu().numpy()
            true = np.argwhere(outputs == 0)
            target = np.argwhere(outputs == 1)
            for i in range(640):
                fake_matrix[true[i][2]][target[i][2]] += 1


# labels: [B, 4]
def get_fake_labels(labels):
    fake_matrix = {'A': 'R', 'B': 'E', 'C': 'Q', 'D': 'P', 'E': 'B', 'F': 'H',
                   'G': 'C', 'H': 'F', 'I': 'l', 'J': 'T', 'K': 'X', 'L': 'I',
                   'M': 'N', 'N': 'M', 'O': '0', 'P': 'D', 'Q': 'G', 'R': 'A',
                   'S': '5', 'T': 'J', 'U': 'V', 'V': 'U', 'W': 'Y', 'X': 'K',
                   'Y': 'W', 'Z': '7', 'a': 'n', 'b': '6', 'c': 'o', 'd': 'P',
                   'e': 'g', 'f': 't', 'g': 'y', 'h': 'r', 'i': 'j', 'j': 'i',
                   'k': 'x', 'l': '1', 'm': 'w', 'n': 'a', 'o': 'c', 'p': 'd',
                   'q': '9', 'r': 'h', 's': '3', 't': 'f', 'u': 'v', 'v': 'u',
                   'w': 'm', 'x': 'k', 'y': 'e', 'z': '2', '0': 'O', '1': 'L',
                   '2': 'z', '3': 's', '4': '8', '5': 'S', '6': 'b', '7': 'Z',
                   '8': '4', '9': 'q'}
    fake_labels = np.zeros(labels.shape)
    for i in range(BATCH):
        for j in range(4):
            fake_labels[i][j] = CHARACTERS.find(fake_matrix[CHARACTERS[labels[i][j]]])
    fake_labels = torch.tensor(fake_labels, dtype=torch.long)
    return fake_labels


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
    acc_t = calc_acc(labels, outputs)
    output_log_softmax = F.log_softmax(outputs, dim=-1)
    input_lengths = torch.tensor([output_log_softmax.shape[0]] * BATCH)
    target_lengths = torch.tensor([4] * BATCH)
    loss_adv = F.ctc_loss(output_log_softmax, get_fake_labels(labels), input_lengths, target_lengths)

    # loss_hinge: bound the magnitude of the perturbation
    C = 0.1
    loss_hinge = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
    loss_hinge = torch.max(loss_hinge - C, torch.zeros(1).to(DEVICE))

    # 损失汇总：加权平均
    ALPHA = 30
    BETA = 20
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


if __name__ == '__main__':
    train_set = CaptchaDataset(TRAIN_SIZE * BATCH)
    train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=12)
    optimizer_d = torch.optim.Adam(D.parameters(), LEARNING_RATE, amsgrad=True)
    optimizer_g = torch.optim.Adam(G.parameters(), LEARNING_RATE, amsgrad=True)
    for epoch in range(1, EPOCHS + 1):
        train(optimizer_d, optimizer_g, epoch, train_loader)
        if epoch > 100 and epoch % 10 == 0:
            torch.save(G.state_dict(), '/home/zhongyu_bishe/code/pipeline/module/GAN_I.pth')

import numpy as np
import cv2
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from steganography.steganalysis.xu import CNN
from gan.train_ocr import decode
from gan.models import CaptchaSolver
from torch.utils.data import DataLoader
from torch.autograd.gradcheck import zero_gradients
from mycaptcha.image import CaptchaDataset, CLASS_NUM, CHARACTERS, WIDTH, HEIGHT, LEN
WHITE_MODEL = 'D'  # 针对白盒模型来产生对抗样本
BLACK_MODEL = 'V'  # 再直接攻击黑盒模型
USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")


def gauss_noise(image, label, model):
    pert_image = copy.deepcopy(image)
    SNR = 5
    for i in range(3):
        c_image = pert_image.cpu().detach().numpy()[0][i]
        noise = np.random.randn(c_image.shape[0], c_image.shape[1])  # 产生N(0,pipeline)噪声数据
        noise = noise - np.mean(noise)
        signal_power = np.linalg.norm(c_image - c_image.mean()) ** 2 / c_image.size
        noise_variance = signal_power / np.power(10, (SNR / 10))
        noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
        pert_image[0][i] = torch.from_numpy(noise + c_image)
    return decode(model(pert_image).permute(1, 0, 2).argmax(dim=-1)[0]) != decode(label), \
           pert_image, \
           decode(model(pert_image).permute(1, 0, 2).argmax(dim=-1)[0])


def fgsm(image, label, model, eps=0.1):
    """
        :param image: [B, C, H, W], B=pipeline
        :param label: true label of image(before decoding)
        :param model: target to attack
        :param max_iter: maximum number of iterations
        :return: a bool value, a perturbed image with [B, C, H, W] and a perturbed label(after decoding)
    """
    assert image.requires_grad
    output = model(image)
    output_log_softmax = F.log_softmax(output, dim=-1)
    loss = F.ctc_loss(output_log_softmax, label, torch.tensor([output_log_softmax.shape[0]]), torch.tensor([4]))
    model.zero_grad()
    loss.backward()
    # torch.clamp(tensor, min, max): element-wise
    # i < min then i = min, i > max then i = max, else i = i
    pert_image = torch.clamp(image + eps*image.grad.data.sign(), 0, 1)
    pert_output = model(pert_image)
    return decode(pert_output.permute(1, 0, 2).argmax(dim=-1)[0]) != decode(label), \
           pert_image,\
           decode(pert_output.permute(1, 0, 2).argmax(dim=-1)[0])


def deepfool(image, model, eps=1e-4, max_iter=70, num_classes=62, overshoot=0.02):
    """
        :param image: [B, C, H, W], B=pipeline
        :param model: target to attack
        :param max_iter: maximum number of iterations
        :param num_classes: num_classes (limits the number of classes to test against, by default = 62)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :return: a bool value, a perturbed image with [B, C, H, W] and a perturbed label
    """
    output = model(image)
    pred_label = decode(output.permute(1, 0, 2).argmax(dim=-1)[0])
    true_label = pred_label
    w = np.zeros(image.shape[3:])
    r_tot = np.zeros(image.shape[3:])
    iter = 0
    # eg.pred_label='2AET',我们只攻击第一个字符，即2，此时t=0
    while pred_label[0] == true_label[0] and iter < max_iter:
        pert_min = 10000
        # 最后一个[i]表示攻击第i个字符
        max = torch.max(output.permute(1, 0, 2)[0], 1)[1][0].item()
        image.retain_grad()
        output[0][0][max].backward(retain_graph=True)
        grad_orig = image.grad.data.cpu().numpy()
        for k in range(0, num_classes):
            if k == max:
                continue
            zero_gradients(image)
            image.retain_grad()
            output[0][0][k].backward(retain_graph=True)
            grad_k = image.grad.data.cpu().numpy()
            w_k = grad_k - grad_orig
            f_k = (output[0][0][k] - output[0][0][max]).data.cpu().numpy()
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            if pert_k < pert_min:
                pert_min = pert_k
                w = w_k
        r_i = (pert_min+eps) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        image[0] = image[0] + (1 + overshoot) * torch.from_numpy(r_tot).to(DEVICE)
        output = model(image)
        pred_label = decode(output.permute(1, 0, 2).argmax(dim=-1)[0])
        iter += 1
    r_tot = (1+overshoot)*r_tot
    return iter < max_iter, image, pred_label


def simba(image, label, model, eps=0.8, max_iter=100):
    """
        :param x: B X C X H X W
        :param y: true label of x
        :return: perturbated x
    """
    # n_dims = C*H*W
    n_dims = torch.prod(torch.tensor(image.shape[1:]))
    # 生成0-n_dims-1随机排列
    # Cartesian basis作为Q的构造标准：Q中元素为长为n_dims的独热码
    perm = torch.randperm(n_dims)
    last_prob = torch.max(torch.nn.Softmax(dim=1)(model(image).cpu().permute(1, 0, 2)[0]), 1)[0]
    for i in range(max_iter):
        diff = torch.zeros(n_dims).to(DEVICE)
        diff[perm[i]] = eps
        neg_image = (image - diff.view(image.size())).clamp(0, 1)
        if decode(model(neg_image).permute(1, 0, 2).argmax(dim=-1)[0]) != decode(label):
            return True, neg_image, decode(model(neg_image).permute(1, 0, 2).argmax(dim=-1)[0])
        neg_prob = torch.max(torch.nn.Softmax(dim=1)(model(neg_image).cpu().permute(1, 0, 2)[0]), 1)[0]
        if torch.sum(neg_prob-last_prob).item() < 1e-5:
            image = neg_image
            last_prob = neg_prob
            continue
        pos_image = (image + diff.view(image.size())).clamp(0, 1)
        if decode(model(pos_image).permute(1, 0, 2).argmax(dim=-1)[0]) != decode(label):
            return True, pos_image, decode(model(pos_image).permute(1, 0, 2).argmax(dim=-1)[0])
        pos_prob = torch.max(torch.nn.Softmax(dim=1)(model(pos_image).cpu().permute(1, 0, 2)[0]), 1)[0]
        if torch.sum(pos_prob-last_prob).item() < 1e-5:
            image = pos_image
            last_prob = pos_prob
    return False, image, decode(label)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def cal_ssim(img1, img2):
    ssims = []
    for i in range(3):
        ssims.append(ssim(img1[0][i].cpu().detach().numpy(), img2[0][i].cpu().detach().numpy()))
    return np.array(ssims).mean()


def cal_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


if __name__ == '__main__':
    test_set = CaptchaDataset(1000*1)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=12)
    white_model = CaptchaSolver(WHITE_MODEL).to(DEVICE)
    white_model.load_state_dict(torch.load("../gan/state_dict/OCR_" + WHITE_MODEL + '.pth', map_location='cpu'))
    white_model.train()
    adv_image = []
    adv_labels = []
    ssim_sum = 0
    for image, labels in test_loader:
        image, labels, = image.to(DEVICE), labels.to(DEVICE)
        image.requires_grad = True
        outputs = white_model(image)
        if not decode(outputs.permute(1, 0, 2).argmax(dim=-1)[0]) == decode(labels[0]):
            continue
        else:
            #attacked, perturbed_image, perturbed_labels = gauss_noise(image, labels[0], white_model)
            #attacked, perturbed_image, perturbed_labels = fgsm(image, labels[0], white_model)
            attacked, perturbed_image, perturbed_labels = simba(image, labels[0], white_model)
            #attacked, perturbed_image, perturbed_labels = deepfool(image, white_model)
            if attacked:
                adv_image.append(perturbed_image)
                adv_labels.append(labels)
                ssim_sum += cal_ssim(image, perturbed_image)
                '''
                if len(adv_image) < 3:
                    transforms\
                        .ToPILImage()(image[0].detach().cpu())\
                        .save('%s_org.png' % decode(labels[0]))
                    transforms\
                        .ToPILImage()(perturbed_image[0].detach().cpu())\
                        .save('%s_per.png' % perturbed_labels)

                '''
                print('WHITE-BOX: Attack successfully:' + decode(labels[0]) + ' -> ' + perturbed_labels)
    print('ssim = %f' % (ssim_sum / len(adv_image)))
    black_model = CaptchaSolver(BLACK_MODEL)
    black_model.load_state_dict(torch.load("../state_dict/TRAINING_" + BLACK_MODEL + '.pth', map_location='cpu'))
    black_model.train()
    black_attack = 0
    for i in range(len(adv_image)):
        image, labels, = adv_image[i].to(DEVICE), adv_labels[i].to(DEVICE)
        outputs = black_model(image)
        if not decode(outputs.permute(1, 0, 2).argmax(dim=-1)[0]) == decode(labels[0]):
            black_attack += 1
            #print('BLACK-BOX: Attack successfully:' +
                #decode(labels[0]) + ' -> ' +
                #decode(outputs.permute(pipeline, 0, 2).argmax(dim=-pipeline)[0]))
    print('%f' % (1-(black_attack / len(adv_image))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from gan.models import CaptchaSolver
from mycaptcha.image import CaptchaDataset, CHARACTERS
BATCH = 128
EPOCHS = 60
LEARNING_RATE = 1e-4
TRAIN_SIZE = 1000
VALID_SIZE = 100
USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
MODEL = 'I'


def decode(sequence):
    # sequence.length = b = lstm_input_size
    a = ''.join([CHARACTERS[x] for x in sequence])
    # 除去'-'和冗余: eg.'-aaaa-2222ddddd-55555' -> 'a2d5'
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != CHARACTERS[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != CHARACTERS[0] and s[-1] != a[-1]:
        s += a[-1]
    return s.upper()


# 计算一批数据网络输出与真实标签的差异
# target.shape = [batch_size, LEN]
# output.shape = [b, batch_size, CLASS_NUM]
def calc_acc(target, output):
    target = target.cpu().numpy()
    # output.shape = [batch_size, b]
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1).cpu().numpy()
    # 注意这里一张图片当且仅当所有字母全部正确才+1，一个batch里再计算平均值
    a = np.array([decode(true) == decode(pred) for true, pred in zip(target, output_argmax)])
    return a.mean()


def train(model, optimizer, epoch, dataloader):
    model.train()
    loss_mean = 0
    acc_mean = 0
    # 进度条
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 优化器梯度清零：防止这个grad同上一个mini-batch有关
            optimizer.zero_grad()
            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            # ctc_loss函数是个大坑:torch的报错有bug不要看
            # output_log_softmax.shape = [b, B, CLASS_NUM]
            # 输入(即之前网络的输出)序列长度,batch size和字符集(这里是characters)总长度
            # target.shape = [B, 4]
            # batch size和标签长度
            # input_lengths = (b, b..., b)，长度为batch size = 128
            # 根据这里，我们在一开始需要定义n_input_length = b = 5
            # target_lengths = (4, 4,..., 4)，长度为batch size = 128
            input_lengths = torch.tensor([output_log_softmax.shape[0]]*BATCH)
            target_lengths = torch.tensor([4] * BATCH)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            loss.backward()
            # 更新参数
            optimizer.step()
            # 提取tensor中的纯Python数值
            loss = loss.item()
            acc = calc_acc(target, output)
            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc
            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean
            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


def valid(model, optimizer, epoch, dataloader):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            input_lengths = torch.tensor([output_log_softmax.shape[0]] * BATCH)
            target_lengths = torch.tensor([4] * BATCH)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            loss = loss.item()
            acc = calc_acc(target, output)
            loss_sum += loss
            acc_sum += acc
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


if __name__ == '__main__':
    train_set = CaptchaDataset(TRAIN_SIZE * BATCH)
    valid_set = CaptchaDataset(VALID_SIZE * BATCH)
    train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=12)
    valid_loader = DataLoader(valid_set, batch_size=BATCH, num_workers=12)
    model = CaptchaSolver(MODEL).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, amsgrad=True)
    for epoch in range(1, EPOCHS + 1):
        train(model, optimizer, epoch, train_loader)
        valid(model, optimizer, epoch, valid_loader)
        if epoch > 20 and epoch % 5 == 0:
            torch.save(model.state_dict(), '/home/zhongyu_bishe/code/gan/state_dict/OCR_' + MODEL + '.pth')


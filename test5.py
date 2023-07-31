# 导入所需的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np

# 定义超参数
batch_size = 32  # 批次大小
num_epochs = 100  # 训练轮数
lr = 0.0002  # 学习率
beta1 = 0.5  # Adam优化器的参数
ngf = 64  # 生成器的特征图数
ndf = 64  # 判别器的特征图数
nc = 1  # 音频通道数
nz = 100  # 噪声维度

# 定义数据集，这里使用torchaudio自带的YesNo数据集作为示例，你可以替换成你自己的数据集
dataset = torchaudio.datasets.YESNO(root='./data', download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# 定义预训练模型，这里使用torchaudio自带的Wav2Vec2模型作为示例，你可以替换成你自己的模型
pretrained_model = torchaudio.models.wav2vec2.Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
pretrained_model.eval()  # 设置为评估模式


# 定义生成器，它将噪声向量转换为音频信号
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是一个nz维的噪声向量，输出是一个4x4xngf*8的特征图
            nn.ConvTranspose1d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # 输入是一个4x4xngf*8的特征图，输出是一个8x8xngf*4的特征图
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # 输入是一个8x8xngf*4的特征图，输出是一个16x16xngf*2的特征图
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # 输入是一个16x16xngf*2的特征图，输出是一个32x32xngf的特征图
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # 输入是一个32x32xngf的特征图，输出是一个64x64xnc的音频信号
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 定义判别器，它将音频信号转换为一个标量，表示该音频是否属于对抗样本
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入是一个64x64xnc的音频信号，输出是一个32x32xndf的特征图
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入是一个32x32xndf的特征图，输出是一个16x16xndf*2的特征图
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入是一个16x16xndf*2的特征图，输出是一个8x8xndf*4的特征图
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入是一个8x8xndf*4的特征图，输出是一个4x4xndf*8的特征图
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入是一个4x4xndf*8的特征图，输出是一个标量
            nn.Conv1d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 创建生成器和判别器
netG = Generator()
netD = Discriminator()

# 定义损失函数，这里使用二元交叉熵损失
criterion = nn.BCELoss()

# 定义优化器，这里使用Adam优化器
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# 定义真假标签
real_label = 1
fake_label = 0

# 开始训练
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # 获取一批音频数据和标签
        audio, label = data

        # 更新判别器
        # 将判别器参数的梯度清零
        netD.zero_grad()
        # 将音频数据输入预训练模型，得到特征向量
        feature = pretrained_model.extract_features(audio)
        # 将特征向量输入判别器，得到真实音频的输出
        output = netD(feature).view(-1)
        # 计算真实音频的损失
        errD_real = criterion(output, torch.full((batch_size,), real_label))
        # 反向传播计算梯度
        errD_real.backward()
        # 计算判别器对真实音频的准确率
        D_x = output.mean().item()

        # 生成一批噪声向量
        noise = torch.randn(batch_size, nz, 1)
        # 将噪声向量输入生成器，得到对抗音频
        fake = netG(noise)
        # 将对抗音频输入判别器，得到对抗音频的输出
        output = netD(fake.detach()).view(-1)
        # 计算对抗音频的损失
        errD_fake = criterion(output, torch.full((batch_size,), fake_label))
        # 反向传播计算梯度
        errD_fake.backward()
        # 计算判别器对对抗音频的准确率
        D_G_z1 = output.mean().item()

        # 计算判别器的总损失和准确率
        errD = errD_real + errD_fake
        D_xz = D_x + D_G_z1

        # 更新判别器参数
        optimizerD.step()

        # 更新生成器
        # 将生成器参数的梯度清零
        netG.zero_grad()
        # 将对抗音频输入判别器，得到对抗音频的输出（注意这里不需要detach）
        output = netD(fake).view(-1)
        # 计算生成器的损失（注意这里的标签是真实标签）
        errG = criterion(output, torch.full((batch_size,), real_label))
        # 反向传播计算梯度
        errG.backward()
        # 计算生成器对对抗音频的准确率
        D_G_z2 = output.mean().item()

        # 更新生成器参数
        optimizerG.step()

        # 打印训练信息
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch + 1, num_epochs, i + 1, len(dataloader),
                     errD.item(), errG.item(), D_xz, D_G_z1, D_G_z2))

        # 保存音频
        if i % 100 == 0:
            torchaudio.save('fake_audio.wav', fake[0], 16000)
            torchaudio.save('real_audio.wav', audio[0], 16000)

    # 保存模型
    torch.save(netG.state_dict(), 'netG.pth')
    torch.save(netD.state_dict(), 'netD.pth')



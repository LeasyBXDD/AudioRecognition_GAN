import torch
import torch.nn as nn
import torchaudio

# 设置音频后端
torchaudio.set_audio_backend("soundfile")


# 定义函数用于计算全连接层的输入维度
def get_num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# 定义模型
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 10))
        self.fc1 = nn.Linear(8 * 10 * 8, 500)  # You may need to adjust these dimensions based on your input
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        # Add an extra dimension to the tensor
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.avgpool(x)
        x = x.view(-1, get_num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # remove sigmoid activation here

        return x


# 创建模型
detector = Detector()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(detector.parameters(), lr=0.001)

# 加载并预处理训练数据
waveform, sample_rate = torchaudio.load('./data/wav48/p226/p226_001.wav')

# 假设我们知道每个样本的长度是1秒
sample_length = sample_rate  # 1秒的样本长度

# 分割波形
samples = waveform[:, :waveform.shape[1] // sample_length * sample_length].split(sample_length, dim=1)

# 对每个非空样本计算梅尔频谱
mel_spectrograms = [torch.squeeze(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(sample)) for sample in
                    samples if sample.nelement() > 0]

# 创建训练数据集和数据加载器
# train_labels = []  # 这应该是你已经定义好的标签列表
train_labels = [0] * 50 + [1] * 50
train_dataset = torch.utils.data.TensorDataset(torch.stack(mel_spectrograms).unsqueeze(1),
                                               torch.tensor(train_labels))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义训练轮数
num_epochs = 10

# 开始训练
for epoch in range(num_epochs):
    total_step = len(train_dataloader)
    for i, (inputs, labels) in enumerate(train_dataloader):
        # 清零梯度
        optimizer.zero_grad()

        # 删除尺寸为1的维度
        inputs = torch.squeeze(inputs)

        # 前向传播
        outputs = detector(inputs)

        # 计算损失
        loss = criterion(outputs, labels.long())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印训练信息
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 预测测试集中的样本
detector.eval()  # 切换到评估模式
with torch.no_grad():  # 不需要计算梯度

    # 加载并预处理测试数据
    waveform, sample_rate = torchaudio.load('./data/wav48/p226/p226_001.wav')

    # 分割波形
    samples = waveform[:, :waveform.shape[1] // sample_length * sample_length].split(sample_length, dim=1)

    # 对每个非空样本计算梅尔频谱
    mel_spectrograms = [torch.squeeze(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(sample)) for sample
                        in samples if sample.nelement() > 0]
    if len(mel_spectrograms) == 0:
        print("No mel spectrograms were generated from the test data. Skipping testing.")
    else:
        # 创建测试数据集和数据加载器
        # test_labels = []  # 这应该是你已经定义好的标签列表
        test_labels = [0] * 50 + [1] * 50
        test_dataset = torch.utils.data.TensorDataset(torch.stack(mel_spectrograms).unsqueeze(1),
                                                      torch.tensor(test_labels))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

        for inputs, labels in test_dataloader:
            # 删除尺寸为1的维度
            inputs = torch.squeeze(inputs)

            outputs = detector(inputs)
            outputs = torch.softmax(outputs, dim=1)  # convert logits to probabilities
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(len(predicted))))

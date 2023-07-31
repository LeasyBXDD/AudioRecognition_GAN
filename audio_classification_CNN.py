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
        self.fc1 = nn.Linear(8 * 10 * 8, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        # Add an extra dimension to the tensor
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.avgpool(x)
        x = x.view(-1, get_num_flat_features(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 加载模型
detector = Detector()
detector.load_state_dict(torch.load('path_to_your_model.pth'))
detector.eval()

# 假设我们知道每个样本的长度是1秒
sample_rate = 16000  # 你需要替换这个为你的实际采样率
sample_length = sample_rate  # 1秒的样本长度


# padding function
def pad_waveform(waveform, desired_length):
    if waveform.size(1) < desired_length:
        num_missing_samples = desired_length - waveform.size(1)
        last_dim_padding = (0, num_missing_samples)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform


# 预测测试集中的样本
with torch.no_grad():  # 不需要计算梯度

    # 音频文件路径列表
    audio_files = ['./audio1.wav', './audio2.wav', './audio3.wav']

    for audio_file in audio_files:

        # 加载并预处理测试数据
        waveform, _ = torchaudio.load(audio_file)

        # 分割波形
        samples = waveform[:, :waveform.shape[1] // sample_length * sample_length].split(sample_length, dim=1)

        # handle the last segment
        if waveform.shape[1] % sample_length != 0:
            last_segment = waveform[:, waveform.shape[1] // sample_length * sample_length:]
            last_segment = pad_waveform(last_segment, sample_length)
            samples.append(last_segment)

        # 对每个非空样本计算梅尔频谱
        mel_spectrograms = [torch.squeeze(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(sample)) for
                            sample
                            in samples if sample.nelement() > 0]
        if len(mel_spectrograms) == 0:
            print("No mel spectrograms were generated from the test data. Skipping testing.")
        else:
            # 将音频样本直接输入到模型中
            inputs = torch.stack(mel_spectrograms).unsqueeze(1)
            inputs = torch.squeeze(inputs)  # 删除尺寸为1的维度

            outputs = detector(inputs)
            outputs = torch.softmax(outputs, dim=1)  # convert logits to probabilities
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            print('Predicted for file {}: '.format(audio_file),
                  ' '.join('%5s' % predicted[j] for j in range(len(predicted))))

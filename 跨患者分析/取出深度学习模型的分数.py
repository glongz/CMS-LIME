import torch
from model.EEGInception_SE import EEGInception

import numpy as np
import os
import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from linear_attention_transformer import LinearAttentionTransformer


from collections import OrderedDict, Counter


def evaluate_accuracy(data_windows, net, device_acc):
    pre_counts = 0
    with torch.no_grad():
        slide_i = 0
        while slide_i < data_windows.shape[0]:
            if isinstance(net, torch.nn.Module):
                X = data_windows[slide_i, :, :]
                slide_i += 1
                net.eval()  # 评估模式, 这会关闭dropout
                X = X.to(torch.float32)
                X = X.unsqueeze(0)
                if X.shape[-1] != 1280:
                    print(X.shape)
                    continue
                predict_cla = net(X.to(device_acc)).argmax(dim=1).cpu().numpy()
                num_of_ones = np.count_nonzero(predict_cla == 1)
                pre_counts += num_of_ones
        # print(data_windows.shape[-1])
        # print('\033[91m________________________________________\033[0m')
        return pre_counts


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sort_key(filename):
    # This regular expression will find the numbers preceded by "FDR" and followed by "-" in the filename
    match = re.search(r'(\d+)-(\d+)', filename)
    if match:
        # Generate a tuple (first_number, second_number)
        return int(match.group(1)), int(match.group(2))
    else:
        return filename


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, freq, time)
        Returns:
            (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)          # -> (batch, time, freq)
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # shape (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_channels: int = 16,
        n_fft: int = 200,
        hop_length: int = 100,
        **kwargs
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        # learnable channel tokens
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.arange(n_channels, dtype=torch.long), requires_grad=False
        )

    # --- STFT ---
    def stft(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: (batch, 1, ts)
        Returns:
            magnitude spec: (batch, freq, time)
        """
        spec = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return spec.abs()

    # --- forward ---
    def forward(self, x: torch.Tensor,
                n_channel_offset: int = 0,
                perturb: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, channel, ts)
        Returns:
            (batch, emb_size)
        """
        emb_seq = []
        B, C, _ = x.shape

        for i in range(C):
            # (B, freq, time)
            spec = self.stft(x[:, i:i + 1, :])
            # (B, time, emb)
            patch_emb = self.patch_embedding(spec)

            # channel token
            token = self.channel_tokens(
                self.index[i + n_channel_offset]
            ).unsqueeze(0).unsqueeze(0)        # (1,1,emb)
            token = token.expand(B, patch_emb.size(1), -1)

            # add & positional encoding
            channel_emb = self.positional_encoding(patch_emb + token)

            # optional random temporal crop (perturb)
            if perturb:
                ts = channel_emb.size(1)
                ts_new = np.random.randint(ts // 2, ts)
                sel = np.random.choice(ts, ts_new, replace=False)
                channel_emb = channel_emb[:, sel]

            emb_seq.append(channel_emb)

        # concat over channels  → (B, C*ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # transformer  → (B, seq, emb) → take mean
        return self.transformer(emb).mean(dim=1)

class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_classes=6, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.biot(x))


class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_channels=18, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor,
                n_channel_offset: int = 0):
        emb1 = self.biot(x, n_channel_offset, perturb=True)
        emb1 = self.prediction(emb1)
        emb2 = self.biot(x, n_channel_offset)
        return emb1, emb2


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate,
                 bn_size, drop_rate, conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('elu1', nn.ELU())
        self.add_module('conv1', nn.Conv1d(num_input_features,
                                           bn_size * growth_rate,
                                           kernel_size=1, stride=1,
                                           bias=conv_bias))
        if batch_norm:
            self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('elu2', nn.ELU())
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3, stride=1,
                                           padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                      p=self.drop_rate,
                                      training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features,
                 bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f'denselayer{i+1}',
                _DenseLayer(num_input_features + i * growth_rate,
                            growth_rate, bn_size, drop_rate,
                            conv_bias, batch_norm))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 conv_bias, batch_norm):
        super().__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('elu', nn.ELU())
        self.add_module('conv', nn.Conv1d(num_input_features,
                                          num_output_features,
                                          kernel_size=1, stride=1,
                                          bias=conv_bias))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):
    def __init__(self, growth_rate=32,
                 block_config=(4, 4, 4, 4, 4, 4, 4),
                 in_channels=16, num_init_features=64,
                 bn_size=4, drop_rate=0.2,
                 conv_bias=True, batch_norm=False):

        super().__init__()

        first_conv = OrderedDict([
            ('conv0', nn.Conv1d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3,
                                bias=conv_bias))
        ])
        if batch_norm:
            first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
        first_conv['elu0'] = nn.ELU()
        first_conv['pool0'] = nn.MaxPool1d(kernel_size=3,
                                           stride=2, padding=1)

        self.densenet = nn.Sequential(first_conv)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.densenet.add_module(
                f'denseblock{i+1}',
                _DenseBlock(num_layers, num_features, bn_size, growth_rate,
                            drop_rate, conv_bias, batch_norm)
            )
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.densenet.add_module(
                    f'transition{i+1}',
                    _Transition(num_features, num_features // 2,
                                conv_bias, batch_norm)
                )
                num_features //= 2

        if batch_norm:
            self.densenet.add_module(f'norm{len(block_config)+1}',
                                     nn.BatchNorm1d(num_features))

        self.densenet.add_module(f'relu{len(block_config)+1}', nn.ReLU())

        # ★★★ 关键修改：固定池化 → 自适应池化 ★★★
        self.densenet.add_module(
            f'pool{len(block_config)+1}',
            nn.AdaptiveAvgPool1d(1)          # 无论输入多短都能输出长度 1
        )

        self.num_features = num_features

        # 官方初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.densenet(x)
        return features.view(features.size(0), -1)   # (B, num_features)


class DenseNetClassifier(nn.Module):
    def __init__(self, growth_rate=32,
                 block_config=(4, 4, 4, 4, 4, 4, 4),
                 in_channels=16, num_init_features=64,
                 bn_size=4, drop_rate=0.2,
                 conv_bias=True, batch_norm=False,
                 drop_fc=0.5, num_classes=6):

        super().__init__()

        self.features = DenseNetEnconder(growth_rate, block_config,
                                         in_channels, num_init_features,
                                         bn_size, drop_rate, conv_bias,
                                         batch_norm)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_fc),
            nn.Linear(self.features.num_features, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


class ResBlock(nn.Module):
    """Convolutional Residual Block 2D
    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.
    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class ContraWR(nn.Module):
    """The encoder model of ContraWR (a supervised model, STFT + 2D CNN layers)
    Yang, Chaoqi, Danica Xiao, M. Brandon Westover, and Jimeng Sun.
    "Self-supervised eeg representation learning for automatic sleep staging."
    arXiv preprint arXiv:2110.15278 (2021).

    @article{yang2021self,
        title={Self-supervised eeg representation learning for automatic sleep staging},
        author={Yang, Chaoqi and Xiao, Danica and Westover, M Brandon and Sun, Jimeng},
        journal={arXiv preprint arXiv:2110.15278},
        year={2021}
    }
    """

    def __init__(self, in_channels=16, n_classes=6, fft=200, steps=20):
        super(ContraWR, self).__init__()
        self.fft = fft
        self.steps = steps
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)

        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(256, n_classes),
        )

    def torch_stft(self, x):
        signal = []
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.fft,
                hop_length=self.fft // self.steps,
                win_length=self.fft,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def forward(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)



def extract_model_scores(patient_w, test_dataset, model_name):
    # model_name = 'eeginception'
    # model_name = 'eegnet'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_model_path = r'D:\public_data\CHBMIT\weight\跨患者微调\%s' % model_name
    # weight_model_path = r'E:\public database\CHBMIT\weight\LOOCV\%s' % model_name
    weight_sub_dirs = os.listdir(weight_model_path)
    weight_sub_dirs = sorted(weight_sub_dirs, key=natural_keys)
    patient = patient_folder = patient_w
    # print("now is patient%d" % patient)
    if patient > 12:
        patient_folder = patient - 1
        print("now is patient%d" % patient)
    elif patient == 12:
        print('exclude 12,now is patient13')
        patient += 1
        patient_folder = patient - 1
    # each_patient_model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_folder - 1])
    # weight_each_patient_sub_dirs = os.listdir(each_patient_model_path)
    # Sort filenames using the sort_key function
    # weight_each_patient_sub_dirs = sorted(weight_each_patient_sub_dirs, key=sort_key)
    # weight_each_patient_sub_dirs = list(filter(lambda x: x.endswith('.pth'), weight_each_patient_sub_dirs))
    # n_seizure = select_loop_n_w - 1
    model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_folder - 1])
    print('model_path is %s' % model_path)
    data_root_dir = r'D:\public_data\CHBMIT\1_data_clean_18channels\chb%02d' % patient
    # ________________________________________________________________________________________________________________________
    data_sub_dirs = os.listdir(data_root_dir)
    data_sub_dirs = list(filter(lambda x: x.endswith('.npy'), data_sub_dirs))
    # ___________________________为了找到通道数___________________________________________________________________________________
    example = data_sub_dirs[0]
    example_sub_patient_dirs = os.path.join(data_root_dir, example)
    example_data = np.load(example_sub_patient_dirs)
    example_data = np.squeeze(example_data)
    n_chans = example_data.shape[0]
    n_classes = 2
    if model_name == 'eeginception':
        from braindecode.models.eegconformer import EEGConformer
        model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)
    elif model_name == 'eegnet':
        from braindecode.models import EEGNetv4
        model = EEGNetv4(n_chans, n_classes, final_conv_length='auto', n_times=1280)
    elif model_name == 'deep4':
        from braindecode.models import Deep4Net
        model = Deep4Net(n_chans=18, n_outputs=2, n_times=1280)
    elif model_name == 'BIOT':
        model = BIOTClassifier(n_fft=200, hop_length=200,depth=4, heads=8,n_channels=n_chans, n_classes=2).to(device)
    elif model_name == 'ATCNet':
        from braindecode.models.atcnet import ATCNet
        model = ATCNet(n_chans=n_chans, n_outputs=2, sfreq=256,input_window_seconds=5)
    elif model_name == 'EEGConformer':
        from braindecode.models.eegconformer import EEGConformer
        model = EEGConformer(n_chans = n_chans, n_outputs = 2, n_times=1280, final_fc_length='auto')
    elif model_name == 'SPaRCNet':
        model = DenseNetClassifier(in_channels=n_chans, num_classes=2).to(device)  #  SPaRCNet
    elif model_name == 'ContraWR':
        model = ContraWR(in_channels=n_chans, n_classes=2, fft=200, steps=20)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    # print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
    counts = list()
    for X, y in tqdm(test_dataset):
        X = X.to(device)
        # X = X.unsqueeze(dim=1)
        # y = y.to(device)
        X = X.to(torch.float32)
        data_counts = evaluate_accuracy(X, net=model, device_acc=device)
        counts.append(data_counts)
    return counts

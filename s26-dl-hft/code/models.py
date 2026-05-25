import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 num_classes=3, dropout=0.2, use_gru=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru

        RNNClass = nn.GRU if use_gru else nn.LSTM
        self.rnn = RNNClass(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


class CausalConv1d(nn.Module):
    # output at t only sees inputs <= t
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    # residual block w/ dilated causal convs, from bai et al 2018
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + self.residual(x))


class TCNClassifier(nn.Module):
    # receptive field grows exponentially with depth
    def __init__(self, input_dim, num_channels=None, kernel_size=3,
                 num_classes=3, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 32, 32, 32]

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        self.num_channels = num_channels

    def forward(self, x):
        out = x.transpose(1, 2)  # TCN wants (batch, channels, seq_len)
        out = self.network(out)
        out = out.mean(dim=2)  # global avg pool
        return self.fc(out)

    def get_receptive_field(self, kernel_size=3):
        n_layers = len(self.num_channels)
        return 1 + 2 * (kernel_size - 1) * (2 ** n_layers - 1)


class PositionalEncoding(nn.Module):
    # sinusoidal, copied from pytorch tutorial basically
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    # attn weights are interpretable which is nice
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, num_classes=3, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._attention_weights = None

    def forward(self, x, return_attention=False):
        out = self.input_proj(x)
        out = self.pos_encoder(out)

        if return_attention:
            attn_weights = self._compute_attention(out)
            self._attention_weights = attn_weights

        out = self.transformer_encoder(out)
        out = out.mean(dim=1)  # no CLS token, just mean pool
        logits = self.fc(out)

        if return_attention:
            return logits, attn_weights
        return logits

    def _compute_attention(self, x):
        # grab attn from first layer for viz
        layer = self.transformer_encoder.layers[0]
        with torch.no_grad():
            q = k = x
            batch_size, seq_len, _ = x.shape
            head_dim = self.d_model // self.nhead

            W = layer.self_attn.in_proj_weight
            b = layer.self_attn.in_proj_bias

            Wq, Wk, Wv = W.chunk(3)
            bq, bk, bv = b.chunk(3)

            Q = F.linear(q, Wq, bq)
            K = F.linear(k, Wk, bk)

            Q = Q.view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.nhead, head_dim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)

        return attn


def create_sequences(features, labels, seq_len=50):
    # sliding window
    T = len(features)
    N = T - seq_len
    if N <= 0:
        raise ValueError(f"Not enough data: T={T}, seq_len={seq_len}")

    X = torch.stack([features[i:i + seq_len] for i in range(N)])
    y = labels[seq_len:]

    return X, y


if __name__ == '__main__':
    batch_size = 16
    seq_len = 50
    input_dim = 13

    x = torch.randn(batch_size, seq_len, input_dim)

    lstm = LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=2)
    out = lstm(x)
    print(f"LSTM output shape: {out.shape}")

    gru = LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=2, use_gru=True)
    out = gru(x)
    print(f"GRU output shape: {out.shape}")

    tcn = TCNClassifier(input_dim=input_dim, num_channels=[16, 16, 16])
    out = tcn(x)
    print(f"TCN output shape: {out.shape}")
    print(f"TCN receptive field: {tcn.get_receptive_field()}")

    transformer = TransformerClassifier(input_dim=input_dim, d_model=32, nhead=4)
    out, attn = transformer(x, return_attention=True)
    print(f"Transformer output shape: {out.shape}")
    print(f"Attention weights shape: {attn.shape}")

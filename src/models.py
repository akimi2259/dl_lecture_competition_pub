import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math

class TransformerEncoderModel(nn.Module):#channel * length
    def __init__(self, num_classes, seq_len, input_size, hidden_size = 64, num_layers = 2):
        super(TransformerEncoderModel, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=0.1)
        
        # Transformerエンコーダーレイヤーを作成
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size, 
                nhead=271, 
                #dtype = torch.float16, 
                batch_first=True
                ),#batch, seq, featureの順
            num_layers=num_layers
        )
        
        # クラス分類用の線形レイヤー
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes),
            nn.Softmax(dim=1)
            )#, dtype = torch.float16)
        
    def forward(self, x):
        #print("x-pos enconder", x)
        x = self.pos_encoder(x)

        #print("x-trans enconder", x)
        # 入力をTransformerエンコーダーに通す
        x = self.transformer_encoder(x)

        #print("x-ave enconder", x)
        
        # 時系列方向の平均を取る
        x = x.mean(dim=1)

        #print("x-class", x)
        
        # クラス分類器を通す
        x = self.classifier(x)

        #print("x-10", x)
        
        return x

"""# 例の使用方法
input_size = 64  # 入力特徴量のサイズ
hidden_size = 128  # Transformerの隠れ層サイズ
num_layers = 2  # Transformerのレイヤー数
num_classes = 10  # 出力クラス数

model = TransformerEncoderModel(input_size, hidden_size, num_layers, num_classes)
print(model)"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 271, dropout: float = 0.1, max_len: int = 281):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin((position * div_term))
        pe[0, :, 1::2] = torch.cos((position * div_term[:-1]))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size,  seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BidirectionalRNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size = 64):
        super(BidirectionalRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Concatenating forward and backward outputs

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

"""# モデルのインスタンス化
input_size = 17781  # 入力の語彙数
hidden_size = 64
output_size = 5  # 分類クラス数
model = BidirectionalRNN(input_size, hidden_size, output_size)
print(model)"""



class BasicConvClassifier(nn.Module):#channel * length
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim1: int = 128,
        hid_dim2: int = 64
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            #ConvBlock(in_channels, in_channels),
            #ConvBlock(in_channels, in_channels),
            ConvBlock(in_channels, hid_dim1),
            #ConvBlock(hid_dim1, hid_dim1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim1, num_classes),
            nn.Dropout(0.3),
            #nn.Dropout(0.2),
            #nn.Linear(hid_dim2, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X) 


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        #self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        nn.init.uniform_(self.conv0.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.conv1.weight, a=-1.0, b=1.0)
        #nn.init.uniform_(self.conv2.weight, a=-1.0, b=1.0)

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        #X = self.conv2(X)
        #X = F.glu(X, dim=-2)

        return self.dropout(X)
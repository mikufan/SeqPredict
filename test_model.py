import torch
import torch.nn as nn
import torch.optim as optim
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * \
                             (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads, num_layers, max_seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.position_encoding(embedded)
        embedded = embedded.permute(1, 0, 2)
        output = self.transformer(embedded, embedded)
        output = output.permute(1, 0, 2)
        logits = self.fc(output[:, -1, :])
        return logits


# 设置模型超参数
vocab_size = 10000  # 词汇表大小
embedding_dim = 256  # 嵌入维度
num_classes = 10  # 类别数
num_heads = 4  # Transformer中的多头注意力头数
num_layers = 4  # Transformer中的编码器和解码器层数
max_seq_len = 100  # 输入序列的最大长度

# 创建模型实例
model = TransformerModel(vocab_size, embedding_dim, num_classes, num_heads, num_layers, max_seq_len)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

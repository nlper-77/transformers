# 参考 https://mp.weixin.qq.com/s/ZllvtpGfkLrcUBKZDtdoTA
import torch
import torch.nn as nn
import os


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # query_weight   (head_dim * head_dim)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # key_weight       (head_dim * head_dim)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # value_weight  (head_dim * head_dim)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N =query.shape[0] # (batch, length, hidden)
        value_len , key_len , query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # (batch, length, head_num, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        

        values = self.values(values) # (batch, length, head_num, head_dim)
        keys = self.keys(keys)
        queries = self.queries(queries)
        

        # 这一步就是计算softmax之前的得分
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys) # 爱因斯坦求和  (batch, head_num, query_length, key_length)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        # 对应的爱因斯坦求和的等价实现
        # import torch
        # q = torch.randn(2,3,4,5)
        # k = torch.randn(2,3,4,5)
        # torch.einsum("nqhd,nkhd->nhqk", q, k) == torch.matmul(q.transpose(1,2), k.transpose(1,2).transpose(3,2))

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))  # Fills elements of self tensor with value where mask is True
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3) # 点积缩放 (batch, head_num, query_length, key_length)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        
        # 对应的爱因斯坦求和等价实现
        # import torch
        # q = torch.randn(2,4,3,3)
        # k = torch.randn(2,3,4,5)
        # torch.einsum("nhql, nlhd->nqhd", [q, k])==torch.matmul(q, k.transpose(1,2)).transpose(1,2)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        # layer norm 对隐状态 比如，512维计算平均值和标准差，然后标准化成服从N(0,1)标准正态分布
        # 参考：https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#layernorm
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)
        # 先升维，再降维
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), 
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attentioned_x = self.attention(value, key, query, mask)
        x = attentioned_x + query  # 残差链接
        下= self.norm1(x)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        x  = forward + x
        x = self.norm2(x)
        out = self.dropout(x)
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size) # (512*768) learning parameter
        # N * transformer_block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # 升维
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # 1) decoder self attention
        attention = self.attention(x, x, x, tgt_mask)
        # 2) Residual, LayerNorm, Dropout
        query = self.dropout(self.norm(attention + x))
        # 3) Encoder-Decoder attention
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
            )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x ,enc_out , src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask) # 训练时并行
        out =self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            tgt_pad_idx,
            embed_size = 256,
            num_layers = 6,
            forward_expansion = 4,
            heads = 8,
            dropout = 0,
            device="cuda",
            max_length=100
        ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
            )
        self.decoder = Decoder(
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
            )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)  -> (batch, head_num, query_length, key_length)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
    tgt = torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)
    src_pad_idx = 0
    tgt_pad_idx = 0
    src_vocab_size = 10
    tgt_vocab_size = 10
    model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, device=device).to(device)
    out = model(x, tgt[:, :-1])
    print(out.shape)
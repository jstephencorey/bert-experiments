import torch.nn as nn
import torch
import numpy as np

# BAsed on https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
# And https://stackoverflow.com/questions/65588829/pytorch-transformer-forward-function-masks-implementation-for-decoder-forward-fu


class Embedding(nn.Module):

    def __init__(self, d_model, vocab_length, sequence_length) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_length, d_model)  # token embedding
        self.pos_embed = nn.Embedding(sequence_length, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)


class EncoderLayer(nn.Module):
   def __init__(self, d_model, feed_forward_dimensions, attention_heads, attention_qkv_dims, dropout):
       super(EncoderLayer, self).__init__()
       self.enc_self_attn = MultiHeadAttention(d_model, attention_heads, attention_qkv_dims)
       self.ffn = FeedForwardNet(d_model, feed_forward_dimensions, dropout)

   def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, feed_forward_dimensions, dropout) -> None:
        super(FeedForwardNet, self).__init__()
        self.expand = nn.Linear(d_model, feed_forward_dimensions)
        self.contract = nn.Linear(feed_forward_dimensions, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        self.dropout(x)
        x = self.expand(x)
        x = self.activation(x)
        self.dropout(x)
        x = self.contract(x)
        x = self.activation(x)
        return self.norm(x)


class MultiHeadAttention(nn.Module):
   def __init__(self, d_model, attention_heads, attention_qkv_dims):
       super(MultiHeadAttention, self).__init__()
       self.W_Q = nn.Linear(d_model, attention_qkv_dims * attention_heads)
       self.W_K = nn.Linear(d_model, attention_qkv_dims * attention_heads)
       self.W_V = nn.Linear(d_model, attention_qkv_dims * attention_heads)
       self.attention_heads = attention_heads
       self.d_model = d_model
       self.attention_qkv_dims = attention_qkv_dims
       self.dot_product_attention = ScaledDotProductAttention(self.attention_qkv_dims)

   def forward(self, Q, K, V, attn_mask):
       # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
       residual, batch_size = Q, Q.size(0)
       # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
       q_s = self.W_Q(Q).view(batch_size, -1, self.attention_heads, self.attention_qkv_dims).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
       k_s = self.W_K(K).view(batch_size, -1, self.attention_heads, self.attention_qkv_dims).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
       v_s = self.W_V(V).view(batch_size, -1, self.attention_heads, self.attention_qkv_dims).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

    #    print("attention heads", self.attention_heads, q_s.size(), k_s.size(), v_s.size(), attn_mask.size())
       attn_mask = attn_mask.unsqueeze(1).repeat(1, self.attention_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

       # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores, context, attn = self.dot_product_attention(q_s, k_s, v_s, attn_mask)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_heads * self.attention_qkv_dims) # context: [batch_size x len_q x n_heads * d_v]
       output = nn.Linear(self.attention_heads * self.attention_qkv_dims, self.d_model)(context)
       return nn.LayerNorm(self.d_model)(output + residual), attn # output: [batch_size x len_q x d_model]


class ScaledDotProductAttention(nn.Module):
   def __init__(self, attention_qkv_dims):
       super(ScaledDotProductAttention, self).__init__()
       self.attention_qkv_dims = attention_qkv_dims
       self.scaling_factor = np.sqrt(self.attention_qkv_dims)
       self.softmax = nn.Softmax(dim=-1)

   def forward(self, Q, K, V,  attn_mask):
       scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scaling_factor # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = self.softmax(scores)
       context = torch.matmul(attn, V)
       return scores, context, attn # Changed from "score, context, attn"


def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class BertModel(nn.Module):

    def __init__(self, d_model = 512, vocab_length = 30, sequence_length = 256, 
                    num_layers = 3, feed_forward_dimensions = 1024, attention_heads = 8,
                    attention_qkv_dims =  128, dropout = 0.1, pad_idx = 3, device = "CPU") -> None:
        super().__init__()
        self.embedding = Embedding(d_model, vocab_length, sequence_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, feed_forward_dimensions, attention_heads, attention_qkv_dims, dropout)
                                for _ in range(num_layers)])
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)

        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))


    def forward(self, x):
        output = self.embedding(x)
        attn_mask = get_attn_pad_mask(x,x)
        for layer in self.encoder_layers:
           output, _ = layer(output, attn_mask)
        # print(output.size())

        output = self.norm(self.activation(self.linear(output)))
        logits_lm = self.decoder(output) + self.decoder_bias

        return logits_lm



def s():
  print("hi")



if __name__ == "__main__":
    bert_model = BertModel()
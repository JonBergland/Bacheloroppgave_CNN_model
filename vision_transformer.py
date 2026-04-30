import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))  

    def forward(self, x):
        return x + self.pos_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout,
            batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 num_classes=10, 
                 embed_dim=768, 
                 num_heads=8, 
                 depth=6, 
                 mlp_dim=None,
                 in_channels=1,
                 dropout=0.1):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 4 * embed_dim
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        seq_len = (img_size // patch_size) ** 2
        self.pos_encoding = PositionalEncoding(embed_dim, seq_len)
        nn.init.trunc_normal_(self.pos_encoding.pos_embed, std=0.02)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(depth)
        ])

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.mlp_head(x[:, 0])
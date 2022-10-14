#nes.py
import torch as th
import torch.nn as nn
from glide_text2im.xf import LayerNorm, Transformer, convert_module_to_f16
from slot_attention import SlotAttention
from positional_encodings.torch_encodings import PositionalEncoding1D

class NES(nn.Module):

  def __init__(self,
        text_ctx=3,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        n_vocab=14,
        model_channels=192,
        xf_padding=True,
        xf_final_ln=True,):
    super(NES, self).__init__()

    self.xf_padding = xf_padding
    self.xf_width = xf_width
    self.num_events = 3
    
    if xf_final_ln:
         self.final_ln = LayerNorm(xf_width)

    if xf_width: 
        self.token_embedding = nn.Embedding(n_vocab, xf_width)
        self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
        self.transformer_proj = nn.Linear(xf_width, model_channels * 4)
        self.p_enc = PositionalEncoding1D(xf_width)

    if self.xf_padding:
        self.padding_embedding = nn.Parameter(
            th.empty(text_ctx, xf_width, dtype=th.float32)
        )
    self.slot_attn = SlotAttention(
            num_slots = self.num_events,
            dim = xf_width,
            iters = 3
        )


  def forward(self, tokens=None, mask=None):
      assert tokens is not None
      xf_in = self.token_embedding(tokens.long())

      pos_enc = self.p_enc(xf_in)
      xf_in = xf_in + pos_enc
      if self.xf_padding:
          assert mask is not None
          xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
      #xf_in: [6,8,512]
      xf_out = self.slot_attn(xf_in)
      Outputs = []
      for i in range(self.num_events):
        embedding = xf_out[:, i]
        xf_proj = self.transformer_proj(embedding)
        outputs = dict(xf_proj=xf_proj, xf_out=embedding.unsqueeze(2))
        Outputs.append(outputs)

      return Outputs
#nes.py

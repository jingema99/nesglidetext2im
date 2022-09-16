#nes.py
import torch as th
import torch.nn as nn
from glide_text2im.xf import LayerNorm, Transformer, convert_module_to_f16

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
        self.transformer = Transformer(
              text_ctx,
              xf_width,
              xf_layers,
              xf_heads,
          )      
        self.token_embedding = nn.Embedding(n_vocab, xf_width)
        self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
        self.transformer_proj = nn.Linear(xf_width, model_channels * 4)

    if self.xf_padding:
        self.padding_embedding = nn.Parameter(
            th.empty(text_ctx, xf_width, dtype=th.float32)
        )



  def forward(self, tokens=None, mask=None):
      assert tokens is not None

      xf_in = self.token_embedding(tokens.long())
      xf_in = xf_in + self.positional_embedding[None]
      if self.xf_padding:
          assert mask is not None
          xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
      xf_out = self.transformer(xf_in)
      if self.final_ln is not None:
          xf_out = self.final_ln(xf_out)
      

      Outputs = []
      xf_proj = self.transformer_proj(xf_out[:, -1])
      xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
      outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
      Outputs.append(outputs)

      return Outputs
#nes.py

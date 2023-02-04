#nes.py
import torch as th
import torch.nn as nn
from glide_text2im.xf import LayerNorm, Transformer, convert_module_to_f16
from glide_text2im.slot_attention import SlotAttention

class NES(nn.Module):

  def __init__(self,
        text_ctx=8,
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
    self.hid_dim = 64
    self.num_events = 3
    
    if xf_width:   
        self.token_embedding = nn.Embedding(n_vocab, self.hid_dim)
        self.positional_embedding = nn.Parameter(th.empty(text_ctx, self.hid_dim, dtype=th.float32))
        self.transformer_proj = nn.Linear(xf_width, model_channels * 4)

    if self.xf_padding:
        self.padding_embedding = nn.Parameter(
            th.empty(text_ctx, self.hid_dim, dtype=th.float32)
        )

    self.conv = nn.Conv1d(self.hid_dim, self.hid_dim, 3, 1, 1)
    self.up = nn.Linear(self.hid_dim, self.xf_width)

    self.slot_attn = SlotAttention(
            num_slots = self.num_events,
            dim = self.xf_width,
            iters = 5
        )

    if xf_final_ln:
         self.final_ln = LayerNorm(xf_width)

  def forward(self, tokens=None, mask=None):
      xf_in = self.token_embedding(tokens.long())
      xf_in = xf_in + self.positional_embedding[None]
      if self.xf_padding:
          assert mask is not None
          xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])

      xf_in = self.conv(xf_in.permute(0,2,1)).permute(0,2,1)
      xf_in = self.up(xf_in)
      event_embedding = self.slot_attn(xf_in)# torch.Size([2, 3, 512])

      Outputs = []
      for event_idx in range(self.num_events):
        embedding = event_embedding[:,event_idx,:] #shape: [2,512]
        xf_proj = self.transformer_proj(embedding)
        outputs = dict(xf_proj=xf_proj, xf_out=embedding.unsqueeze(2))
        Outputs.append(outputs)
      return Outputs


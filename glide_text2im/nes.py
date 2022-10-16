#nes.py
import torch as th
import torch.nn as nn
from glide_text2im.xf import LayerNorm, Transformer, convert_module_to_f16
from slot_attention import SlotAttention

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
    self.win_size = 4
    
    if xf_final_ln:
         self.final_ln = LayerNorm(xf_width)

    if xf_width: 
        # self.transformer = Transformer(
        #     text_ctx,
        #     xf_width,
        #     xf_layers,
        #     xf_heads,
        # )
        self.token_embedding = nn.Embedding(n_vocab, xf_width)
        self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
        self.transformer_proj = nn.Linear(xf_width, model_channels * 4)

    if self.xf_padding:
        self.padding_embedding = nn.Parameter(
            th.empty(text_ctx, xf_width, dtype=th.float32)
        )
    self.slot_attn = SlotAttention(
            num_slots = self.num_events,
            dim = xf_width,
            iters = 3
        )

    device = th.device("cuda:0")
    self.w_transformer1 = Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device)                        
    self.sw_transformer1 = [Transformer(self.win_size/2,xf_width,xf_layers,xf_heads).to(device),
                           Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device)]
    self.w_transformer2 = Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device)                        
    self.sw_transformer2 = [Transformer(self.win_size/2,xf_width,xf_layers,xf_heads).to(device),
                           Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device)]
    # self.sw_transformer = [Transformer(self.win_size/2,xf_width,xf_layers,xf_heads).to(device),
    #                        Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device),
    #                        Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device),
    #                        Transformer(self.win_size/2,xf_width,xf_layers,xf_heads).to(device),]

    self.out_transformer = Transformer(self.win_size,xf_width,xf_layers,xf_heads).to(device)


  def win_transformer(self, xf_in, size, transformer):
    
    num_windows = int(xf_in.shape[1]/size)
    xf_out = xf_in.new_zeros(xf_in.shape)
    start = 0
    for i in range(num_windows):
      step = size
      xf_window = xf_in[:,start:start+step,:]
      xf_out[:,start:start+step,:] = transformer(xf_window)
      start += step
      
    return xf_out
  
  def swin_transformer(self, xf_in, size, transformer):

    num_windows = int(xf_in.shape[1]/size) + 1
    xf_out = xf_in.new_zeros(xf_in.shape)
    
    start = 0
    for i in range(num_windows):
      if i == 0 or i == num_windows-1:
        step = int(size/2)
      else:
        step = size
      xf_window = xf_in[:,start:start+step,:]
      if  i == 0 or i == num_windows-1:
        xf_out[:,start:start+step,:] = transformer[0](xf_window)
      else:
        xf_out[:,start:start+step,:] = transformer[1](xf_window)
      start += step

    return xf_out

  def out(self, xf_in, size, transformer):
    
    num_windows = int(xf_in.shape[1]/size)
    xf_outs = []

    start = 0
    for i in range(num_windows):
      step = size
      xf_window = xf_in[:,start:start+step,:]
      xf_outs.append(transformer(xf_window)[:,-1])
      start += step
      
    return xf_outs


  def forward(self, tokens=None, mask=None):
      assert tokens is not None

      xf_in = self.token_embedding(tokens.long())
      if self.xf_padding:
          assert mask is not None
          xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])

      #xf_in [B, 12, 9]
      xf_in = self.win_transformer(xf_in, self.win_size, self.w_transformer1)
      xf_in = self.swin_transformer(xf_in, self.win_size, self.sw_transformer1)
      xf_in = self.win_transformer(xf_in, self.win_size, self.w_transformer2)
      xf_in = self.swin_transformer(xf_in, self.win_size, self.sw_transformer2)

      xf_outs = self.out(xf_in, self.win_size, self.out_transformer)

      Outputs = []
      for i in range(len(xf_outs)):
        embedding = xf_outs[i]
        xf_proj = self.transformer_proj(embedding)
        outputs = dict(xf_proj=xf_proj, xf_out=embedding.unsqueeze(2))
        Outputs.append(outputs)

      return Outputs
#nes.py

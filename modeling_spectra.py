"""SPECTRA self-contained HF model: latent-KV Llama with a REAL packed-int4 latent KV cache.
AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True) loads it; model.generate() runs Mode A
(store packed int4 latent -> reconstruct K,V + exact RoPE -> attend), so KV memory is at the compression
rate (~7-8x), not fp16; decode is slower than dense because K and V are rebuilt at every step.
k_proj/v_proj are low-rank down-projections (W_DOWN); o_proj is closed-form healed."""
import torch, torch.nn as nn, types, math
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

class LatentProj(nn.Module):
    def __init__(self, in_dim, r, out_dim, bias=False):
        super().__init__()
        self.W_DOWN=nn.Parameter(torch.zeros(in_dim, r))
        self.W_UP=nn.Parameter(torch.zeros(r, out_dim))
        self.mu=nn.Parameter(torch.zeros(r)); self.beta=nn.Parameter(torch.zeros(out_dim))
    def latent(self, x): return x.to(self.W_DOWN.dtype)@self.W_DOWN - self.mu
    def forward(self, x): return self.latent(x)@self.W_UP + self.beta

def _pack_i4(c):
    B,T,r=c.shape; h=r//2
    amax=c.abs().amax(-1,keepdim=True).clamp_min(1e-6); s=amax/7.0
    q=(c/s).round().clamp(-8,7).to(torch.int32)&0xF
    return (q[...,:h]|(q[...,h:]<<4)).to(torch.uint8), s.to(torch.float16)
def _unpack_i4(p,s):
    lo=((p&0xF).to(torch.int32)^8)-8; hi=(((p>>4)&0xF).to(torch.int32)^8)-8
    return torch.cat([lo,hi],-1).to(torch.float16)*s

class _Cache:
    def __init__(self): self.k={};self.ks={};self.v={};self.vs={}
    def reset(self): self.k={};self.ks={};self.v={};self.vs={}

def _spectra_attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                          past_key_value=None, past_key_values=None, cache_position=None, **kw):
    B,T,_=hidden_states.shape; x=hidden_states; li=self._li; C=self._cache
    nq=self.config.num_attention_heads; nkv=self.config.num_key_value_heads; d=self.head_dim
    if li==0 and T>1: C.reset()   # new prefill => new sequence
    q=self.q_proj(x).view(B,T,nq,d).transpose(1,2)
    ck=self.k_proj.latent(x); cv=self.v_proj.latent(x)
    pk,sk=_pack_i4(ck.float()); pv,sv=_pack_i4(cv.float())
    if li not in C.k: C.k[li]=pk;C.ks[li]=sk;C.v[li]=pv;C.vs[li]=sv
    else:
        C.k[li]=torch.cat([C.k[li],pk],1);C.ks[li]=torch.cat([C.ks[li],sk],1)
        C.v[li]=torch.cat([C.v[li],pv],1);C.vs[li]=torch.cat([C.vs[li],sv],1)
    Tc=C.k[li].shape[1]
    ckf=_unpack_i4(C.k[li],C.ks[li]); cvf=_unpack_i4(C.v[li],C.vs[li])
    K=(ckf.to(self.k_proj.W_UP.dtype)@self.k_proj.W_UP+self.k_proj.beta).view(B,Tc,nkv,d).transpose(1,2)
    V=(cvf.to(self.v_proj.W_UP.dtype)@self.v_proj.W_UP+self.v_proj.beta).view(B,Tc,nkv,d).transpose(1,2)
    pos=torch.arange(Tc,device=x.device)[None]
    cos,sin=self._rot(K.float(),pos)
    _,K=apply_rotary_pos_emb(K,K,cos,sin)  # RoPE K (dummy q slot)
    q2,_=apply_rotary_pos_emb(q,q,cos[:,Tc-T:Tc],sin[:,Tc-T:Tc])
    K=repeat_kv(K,nq//nkv); V=repeat_kv(V,nq//nkv)
    o=F.scaled_dot_product_attention(q2.to(V.dtype),K.to(V.dtype),V,is_causal=(T>1))
    o=o.transpose(1,2).reshape(B,T,-1).to(hidden_states.dtype)
    return self.o_proj(o), None

class SpectraLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        rk=int(getattr(config,"spectra_r_k",512)); rv=int(getattr(config,"spectra_r_v",512))
        H=config.hidden_size; nkv=config.num_key_value_heads
        hd=getattr(config,"head_dim",H//config.num_attention_heads); kvd=nkv*hd
        cache=_Cache()
        for i,l in enumerate(self.model.layers):
            sa=l.self_attn
            sa.k_proj=LatentProj(H,rk,kvd); sa.v_proj=LatentProj(H,rv,kvd)
            sa._li=i; sa._cache=cache; sa._rot=self.model.rotary_emb
            sa.forward=types.MethodType(_spectra_attn_forward,sa)
        self.config.use_cache=False

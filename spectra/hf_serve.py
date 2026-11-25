"""Real-memory HF serving for SPECTRA: a KV cache that stores the PACKED int4 low-rank LATENT (not fp16
K/V), so model.generate() uses memory at the true compression rate. Mode A: reconstruct K,V from the latent
+ exact RoPE at read (slower than dense attention: this path saves memory, not time). Enable on a
spectra.compress(..., preset="vllm-7x") model via enable(model, cache). Llama-family models."""
import torch, types, math
import torch.nn.functional as F

def _pack_i4(c):  # c [B,T,r] -> packed uint8 [B,T,r//2], scale fp16 [B,T,1]  (half-split, per-token symmetric)
    B,T,r=c.shape; h=r//2
    amax=c.abs().amax(-1,keepdim=True).clamp_min(1e-6); s=(amax/7.0)
    q=(c/s).round().clamp(-8,7).to(torch.int32)&0xF
    packed=(q[...,:h]|(q[...,h:]<<4)).to(torch.uint8)
    return packed, s.to(torch.float16)
def _unpack_i4(packed, s):  # -> c [B,T,r]
    lo=((packed&0xF).to(torch.int32)^8)-8; hi=(((packed>>4)&0xF).to(torch.int32)^8)-8
    return torch.cat([lo,hi],-1).to(torch.float16)*s

class SpectraCache:
    """Per-layer packed-int4 latent store. Holds K,V latent (compressed) instead of fp16 K,V."""
    def __init__(self): self.k={}; self.v={}; self.ks={}; self.vs={}; self.n=0
    def reset(self): self.__init__()

def enable(model, cache):
    """Patch every LlamaAttention to (down->pack->store latent) + (unpack->reconstruct->RoPE->attend)."""
    dev=next(model.parameters()).device
    rot=model.model.rotary_emb
    layers=model.model.layers
    for li,l in enumerate(layers): l.self_attn._li=li; l.self_attn._cache=cache; l.self_attn._rot=rot
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_value=None, past_key_values=None, **kw):
        B,T,_=hidden_states.shape; x=hidden_states; li=self._li; C=self._cache
        nq=self.config.num_attention_heads; nkv=self.config.num_key_value_heads; d=self.head_dim
        q=self.q_proj(x).view(B,T,nq,d).transpose(1,2)
        kp=self.k_proj; vp=self.v_proj  # LatentProjV2
        ck=(x.to(kp.W_DOWN.dtype)@kp.W_DOWN-kp.mu)   # [B,T,r_k]
        cv=(x.to(vp.W_DOWN.dtype)@vp.W_DOWN-vp.mu)
        pk,sk=_pack_i4(ck.float()); pv,sv=_pack_i4(cv.float())
        if li not in C.k: C.k[li]=pk; C.ks[li]=sk; C.v[li]=pv; C.vs[li]=sv
        else:
            C.k[li]=torch.cat([C.k[li],pk],1); C.ks[li]=torch.cat([C.ks[li],sk],1)
            C.v[li]=torch.cat([C.v[li],pv],1); C.vs[li]=torch.cat([C.vs[li],sv],1)
        Tc=C.k[li].shape[1]
        ckf=_unpack_i4(C.k[li],C.ks[li]); cvf=_unpack_i4(C.v[li],C.vs[li])
        K=(ckf.to(kp.W_UP.dtype)@kp.W_UP+kp.beta).view(B,Tc,nkv,d).transpose(1,2)
        V=(cvf.to(vp.W_UP.dtype)@vp.W_UP+vp.beta).view(B,Tc,nkv,d).transpose(1,2)
        # RoPE: q at current positions [Tc-T..Tc), K at all [0..Tc)
        pos=torch.arange(Tc,device=x.device)
        cos,sin=self._rot(K, pos[None])
        def rope(t,c,s):
            def rh(u): h=u.shape[-1]//2; return torch.cat([-u[...,h:],u[...,:h]],-1)
            return t*c.unsqueeze(1)+rh(t)*s.unsqueeze(1)
        K=rope(K,cos,sin)
        qcos=cos[:,Tc-T:Tc]; qsin=sin[:,Tc-T:Tc]
        q=rope(q,qcos,qsin)
        g=nq//nkv
        K=K.repeat_interleave(g,1); V=V.repeat_interleave(g,1)
        o=F.scaled_dot_product_attention(q.to(V.dtype),K,V,is_causal=(T>1))
        o=o.transpose(1,2).reshape(B,T,-1).to(hidden_states.dtype)
        return self.o_proj(o), None
    for l in layers: l.self_attn.forward=types.MethodType(fwd,l.self_attn)
    model.config.use_cache=False
    return model

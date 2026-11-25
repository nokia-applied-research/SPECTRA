"""Fused SPECTRA-Flash decode kernels vs PyTorch references (CUDA + Triton; fp8 needs sm89+).

    python tests/test_flash_kernels.py
"""
import torch


def test_kernels():
    from spectra.flash.kernels import q_prep, rec_write, tail_merge
    torch.manual_seed(0)
    dev = "cuda"; BF, F8T = torch.bfloat16, torch.float8_e4m3fn
    H, HKV, G, D, R, r = 32, 8, 4, 128, 512, 64
    for B in (1, 3, 16):
        NQ = 6144 + H * r + R + r
        qkvr = (torch.randn(B, NQ, device=dev) * 0.5).to(BF)
        wuk = (torch.randn(HKV, D, R, device=dev) * 0.05).to(BF)
        wuv = (torch.randn(HKV, R, D, device=dev) * 0.05).to(BF)
        pos = torch.randint(30000, 33000, (B,), device=dev)
        theta = torch.rand(r // 2, device=dev) * 0.5
        ss = torch.rand(r, device=dev) + 0.5
        qs = torch.tensor([0.02], device=dev)
        ang = pos.float()[:, None, None] * theta[None, None]
        def rp(x):
            x2 = x.float().unflatten(-1, (-1, 2)); a, c_ = x2[..., 0], x2[..., 1]
            return torch.stack([a * ang.cos() - c_ * ang.sin(), c_ * ang.cos() + a * ang.sin()], -1).flatten(-2)
        qa = torch.einsum("bgjd,gdr->bgjr", qkvr[:, :4096].float().view(B, HKV, G, D), wuk.float()).reshape(B, H, R)
        qr = rp(qkvr[:, 6144:6144 + H * r].view(B, H, r))
        for f8 in (False, True):
            out = torch.empty(B, H, R + r, device=dev, dtype=F8T if f8 else BF)
            q_prep(qkvr, wuk, pos, theta, ss, qs, out, f8)
            ref = torch.cat([qa, qr * (ss if f8 else 1)], -1) / (qs if f8 else 1)
            ref = ref.clamp(-448, 448).to(F8T).float() if f8 else ref
            e = ((out.float() - ref).norm() / ref.norm()).item()
            print(f"[kernels] q_prep B={B} f8={f8}: rel err {e:.2e}"); assert e < (3e-2 if f8 else 1e-2), (B, f8, e)
        # tail_merge over a ring (sink 4 + ring 124): positions in the ring at slot q (q < 4) or 4 + q % 124, the slots
        # not written yet masked; both cache layouts (engine / HF: K and V tensors; vLLM: K|V per head in ring blocks)
        SINK, RING = 4, 124
        CAP = SINK + RING
        qf = qkvr[:, :4096]
        pos_w = torch.tensor(([2, 50, 123, 124, 126, 127, 30001] * 3)[:B], device=dev)
        rlen = torch.where(pos_w >= SINK + RING, pos_w - CAP + 1, 0).to(torch.int32)
        o_l = (torch.randn(B, 1, H, R, device=dev) * 0.5).to(BF)
        o_l[rlen == 0] = float("nan")                      # must be ignored: no record entries yet
        lse_r = torch.randn(B, H, 1, device=dev) * 2 + 5
        j = torch.arange(CAP, device=dev)
        q0 = torch.where(j - SINK < SINK, j - SINK + RING, j - SINK)
        valid = torch.where(j[None] < SINK, j[None] <= pos_w[:, None], q0[None] <= pos_w[:, None])   # (B, CAP)
        Kr = (torch.randn(B, CAP, HKV, D, device=dev) * 0.5).to(BF); Vr = torch.randn_like(Kr)
        refo = torch.empty(B, H, D, device=dev)
        for bb in range(B):
            qq = qf[bb].float().view(HKV, G, D)
            Kb, Vb = Kr[bb, valid[bb]].float(), Vr[bb, valid[bb]].float()
            sc = torch.einsum("gjd,tgd->gjt", qq, Kb) * D ** -0.5
            oe = torch.einsum("gjt,tgd->gjd", sc.softmax(-1), Vb).reshape(H, D)
            if rlen[bb] > 0:
                lse_e = torch.logsumexp(sc, -1).reshape(H)
                orr = torch.einsum("gjr,grd->gjd", o_l[bb, 0].float().view(HKV, G, R), wuv.float()).reshape(H, D)
                w = torch.sigmoid(lse_r[bb, :, 0] - lse_e)[:, None]
                refo[bb] = orr * w + oe * (1 - w)
            else:
                refo[bb] = oe
        SPS = 128                                           # engine / HF: B * SPS slots, K and V separate
        tk = torch.zeros(B * SPS, HKV, D, device=dev, dtype=BF); tv = torch.zeros_like(tk)
        tk.view(B, SPS, HKV, D)[:, :CAP] = Kr; tv.view(B, SPS, HKV, D)[:, :CAP] = Vr
        out = torch.empty(B, H * D, device=dev, dtype=BF)
        tail_merge(qf, qkvr.stride(0), tk, tv, torch.arange(B, device=dev) * SPS, pos_w, rlen, o_l, lse_r, wuv, out,
                   D ** -0.5, SINK, RING)
        e = ((out.float().view(B, H, D) - refo).norm() / refo.norm()).item()
        print(f"[kernels] tail_merge B={B} (separate K / V): rel err {e:.2e}"); assert e < 1e-2, (B, e)
        CW, NB = 135, 2 * B + 3                             # vLLM: ring blocks of 135 slots, K|V interleaved per head
        kv = torch.randn(NB, CW, HKV, 2 * D, device=dev).to(BF)
        blocks = torch.randperm(NB, device=dev)[:B]
        kv[blocks, :CAP, :, :D] = Kr; kv[blocks, :CAP, :, D:] = Vr
        flat = kv.view(NB * CW, HKV, 2 * D)
        out2 = torch.empty_like(out)
        tail_merge(qf, qkvr.stride(0), flat[..., :D], flat[..., D:], blocks * CW, pos_w, rlen, o_l, lse_r, wuv, out2,
                   D ** -0.5, SINK, RING)
        e2 = ((out2.float().view(B, H, D) - refo).norm() / refo.norm()).item()
        print(f"[kernels] tail_merge B={B} (vLLM ring blocks): rel err {e2:.2e}, max |diff| vs separate "
              f"{(out2.float() - out.float()).abs().max().item():.1e}")
        assert e2 < 1e-2 and torch.equal(out2, out), (B, e2)
        # rec_write
        e0 = qkvr[:, 6144 + H * r:]                                   # a strided row view
        e1 = (torch.randn(B, R + r, device=dev) * 0.5).to(BF)          # a contiguous buffer (other row stride)
        for f8 in (False, True):
            rec = torch.zeros(B * 4, R + r, device=dev, dtype=F8T if f8 else BF)
            slot = torch.arange(B, device=dev) * 4 + 1
            skip = slot.clone(); skip[::2] = -1                        # -1 = no entry: nothing written
            rec_write(e0, e1, pos, theta, rec, skip, f8)
            assert int((rec.float().sum(-1) != 0).sum()) == int((skip >= 0).sum()), "rec_write wrote a skipped row"
            rec.zero_()
            rec_write(e0, e1, pos, theta, rec, slot, f8)
            ent = e0.float() + e1.float()
            ang1 = pos.float()[:, None] * theta[None]
            x2 = ent[:, R:].unflatten(-1, (-1, 2)); a, c_ = x2[..., 0], x2[..., 1]
            refe = torch.cat([ent[:, :R], torch.stack([a * ang1.cos() - c_ * ang1.sin(), c_ * ang1.cos() + a * ang1.sin()], -1).flatten(-2)], -1)
            refe = refe.clamp(-448, 448).to(F8T).float() if f8 else refe
            got = rec[slot].float()
            err = ((got - refe).norm() / refe.norm()).item()
            print(f"[kernels] rec_write B={B} f8={f8}: rel err {err:.2e}"); assert err < (3e-2 if f8 else 1e-2), (B, f8, err)
            assert int((rec.float().sum(-1) != 0).sum()) == B, "rec_write touched other slots"
            # one layer per record (n = 1, SPECTRA-Flash): the entry is e0 alone
            rec.zero_()
            rec_write(e0, None, pos, theta, rec, slot, f8)
            ent1 = e0.float()
            x1 = ent1[:, R:].unflatten(-1, (-1, 2)); a1, c1 = x1[..., 0], x1[..., 1]
            ref1 = torch.cat([ent1[:, :R], torch.stack([a1 * ang1.cos() - c1 * ang1.sin(), c1 * ang1.cos() + a1 * ang1.sin()], -1).flatten(-2)], -1)
            ref1 = ref1.clamp(-448, 448).to(F8T).float() if f8 else ref1
            err1 = ((rec[slot].float() - ref1).norm() / ref1.norm()).item()
            print(f"[kernels] rec_write n=1 B={B} f8={f8}: rel err {err1:.2e}"); assert err1 < (3e-2 if f8 else 1e-2), (B, f8, err1)



if __name__ == "__main__":
    test_kernels()
    print("[kernels] all passed")

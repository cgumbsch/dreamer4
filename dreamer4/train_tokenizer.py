# train_tokenizer.py
import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

import wandb

from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    recon_loss_from_mae, lpips_on_mae_recon,
    EmaRms,
)

try:
    import lpips
except ImportError:
    lpips = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


@torch.no_grad()
def log_tokenizer_viz_wandb(
    *,
    x_btchw: torch.Tensor,          # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,        # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,   # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "tokenizer/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    # patchify target
    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)  # (B,Tv,Np,Dp)

    # panels (patch space)
    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    # to image space (B,T,C,H,W)
    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,H,T*W)
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)  # (Bv,C,4H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*4H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {
            tag: wandb.Image(
                big_hwc,
                caption="rows=target/masked/recon_masked/recon_full",
            ),
            "tokenizer/masked_frac": float(mae_mask_btNp1[:, :Tv].float().mean().item()),
        },
        step=step,
    )


def save_ckpt(path: Path, *, step: int, epoch: int, model, opt, args: argparse.Namespace, rms=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": opt.state_dict(),
        "args": vars(args),
        "scale_pos_embeds": args.scale_pos_embeds,
    }
    if rms is not None:
        obj["rms"] = {k: v.state_dict() for k, v in rms.items()}
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, model, opt, rms=None) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=True)
    opt.load_state_dict(ckpt["opt"])
    if rms is not None and ckpt.get("rms") is not None:
        for k, v in rms.items():
            if k in ckpt["rms"]:
                v.load_state_dict(ckpt["rms"][k])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    # ---- data ----
    dataset = ShardedFrameDataset(
        outdirs=args.data_dirs,
        tasks=TASK_SET,
        seq_len=args.seq_len,
        iid_sampling=True,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    # ---- model ----
    assert args.H % args.patch == 0 and args.W % args.patch == 0
    n_patches = (args.H // args.patch) * (args.W // args.patch)
    d_patch = args.patch * args.patch * args.C

    assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    enc = Encoder(
        patch_dim=d_patch,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_patches=n_patches,
        n_heads=args.n_heads,
        depth=args.depth,
        d_bottleneck=args.d_bottleneck,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        mae_p_min=args.mae_p_min,
        mae_p_max=args.mae_p_max,
        scale_pos_embeds=args.scale_pos_embeds,
        qk_norm=args.qk_norm,
        attn_softcap=args.attn_softcap,
        rope=args.rope,
    )
    dec = Decoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
        qk_norm=args.qk_norm,
        attn_softcap=args.attn_softcap,
        rope=args.rope,
    )
    model = Tokenizer(enc, dec, pos_offset_max=args.pos_offset_max).to(device)

    if is_rank0():
        print(model)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # ---- optim ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = torch.cuda.is_available()

    rms_mse = EmaRms().to(device)
    rms_lp = EmaRms().to(device)
    rms = {"mse": rms_mse, "lp": rms_lp}

    # ---- lpips ----
    if args.lpips_weight > 0.0:
        assert lpips is not None, "pip install lpips"
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval()
        lpips_fn.requires_grad_(False)
    else:
        lpips_fn = None

    # ---- wandb ----
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online",
            config=vars(args),
        )

    # ---- resume ----
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), model=model, opt=opt, rms=rms)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # ---- train ----
    model.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    epoch = start_epoch  # bound even if the while-body is skipped (resume at/after max_steps)
    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for x in loader:
                if step >= args.max_steps:
                    break

                x = x.to(device, non_blocking=True)  # (B,T,C,H,W)
                patches = temporal_patchify(x, args.patch)

                with torch.no_grad():
                    if is_rank0() and step % args.log_every == 0:
                        z, _ = (model.module.encoder if hasattr(model, "module") else model.encoder)(patches)
                        zf = z.float()
                        z_spread = float(zf.flatten(1).std(dim=0).mean().item())
                        wandb.log({
                            "debug/z_std": float(zf.std().item()),
                            "debug/z_mean": float(zf.mean().item()),
                            "debug/z_abs_max": float(zf.abs().max().item()),
                            "debug/z_spread": z_spread,
                        }, step=step)
                        if step >= args.collapse_check_after and z_spread < args.collapse_spread_min:
                            raise RuntimeError(
                                f"Bottleneck collapse: between-sample z_spread={z_spread:.2e} "
                                f"< {args.collapse_spread_min} at step {step}"
                            )

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred, mae_mask, keep_prob = model(patches)

                # losses in fp32 (outside autocast)
                mse = recon_loss_from_mae(pred, patches, mae_mask)
                rms_mse.update(mse)

                if lpips_fn is not None and args.lpips_weight > 0.0:
                    lp = lpips_on_mae_recon(
                        lpips_fn, pred, patches, mae_mask,
                        H=args.H, W=args.W, C=args.C, patch=args.patch,
                        subsample_frac=args.lpips_frac
                    )
                    rms_lp.update(lp)
                    loss = rms_mse.normalize(mse) + args.lpips_weight * rms_lp.normalize(lp)
                else:
                    lp = torch.zeros((), device=device)
                    loss = rms_mse.normalize(mse)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss} mse={mse} lp={lp}")

                loss_to_backprop = loss / grad_accum

                loss_to_backprop.backward()

                do_step = ((step + 1) % grad_accum == 0)
                grad_norm_enc = grad_norm_dec = 0.0
                if do_step:
                    _model = model.module if hasattr(model, "module") else model
                    clip_enc = args.grad_clip_enc if args.grad_clip_enc > 0 else float("inf")
                    clip_dec = args.grad_clip_dec if args.grad_clip_dec > 0 else float("inf")
                    grad_norm_enc = float(torch.nn.utils.clip_grad_norm_(
                        _model.encoder.parameters(), max_norm=clip_enc).item())
                    grad_norm_dec = float(torch.nn.utils.clip_grad_norm_(
                        _model.decoder.parameters(), max_norm=clip_dec).item())

                    if args.warmup_steps > 0 and step < args.warmup_steps:
                        warmup_frac = (step + 1) / args.warmup_steps
                        for pg in opt.param_groups:
                            pg["lr"] = args.lr * warmup_frac

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # ---- logging ----
                if is_rank0() and (step % args.log_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    _model = model.module if hasattr(model, "module") else model
                    weight_norm_enc = float(sum(p.float().norm().item() ** 2 for p in _model.encoder.parameters()) ** 0.5)
                    weight_norm_dec = float(sum(p.float().norm().item() ** 2 for p in _model.decoder.parameters()) ** 0.5)
                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/mse": float(mse.item()),
                            "loss/lpips": float(lp.item()),
                            "stats/psnr": float(psnr.item()),
                            "stats/keep_prob": float(keep_prob.mean().item()),
                            "stats/masked_frac": float(mae_mask.float().mean().item()),
                            "stats/grad_norm_enc": grad_norm_enc,
                            "stats/grad_norm_dec": grad_norm_dec,
                            "stats/weight_norm_enc": weight_norm_enc,
                            "stats/weight_norm_dec": weight_norm_dec,
                            "rms/mse": rms_mse.rms_val,
                            "rms/lp": rms_lp.rms_val,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        },
                        step=step,
                    )

                if is_rank0() and (step % args.print_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mse={mse.item():.6f} | lpips={lp.item():.4f} "
                        f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f}"
                    )

                # ---- viz ----
                if is_rank0() and args.viz_every > 0 and (step % args.viz_every == 0):
                    log_tokenizer_viz_wandb(
                        x_btchw=x,
                        pred_btnd=pred,
                        mae_mask_btNp1=mae_mask,
                        patch=args.patch,
                        step=step,
                        max_items=args.viz_max_items,
                        max_T=args.viz_max_T,
                    )

                # ---- ckpt ----
                # save on optimizer boundaries (compatible with grad_accum > 1)
                if is_rank0() and args.save_every > 0 and do_step and (step % args.save_every) < grad_accum:
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, model=model, opt=opt, args=args, rms=rms)
                    # also update a "latest" pointer
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, model=model, opt=opt, args=args, rms=rms)

                step += 1

            start_epoch = epoch + 1
            if step >= args.max_steps:
                # The inner loader-loop already broke at max_steps; without this the outer
                # epoch-loop (range up to 10M) keeps fetching one batch and breaking forever,
                # pinning the GPU at 0% util and never re-checking the `while` guard. Break so
                # training self-exits at max_steps.
                break

    # Always persist a final checkpoint at the last processed step, regardless of save_every
    # alignment, so latest.pt matches the exact step we intend to freeze (the in-loop save
    # fires only on step % save_every == 0 and never lands on max_steps).
    if is_rank0():
        final_ckpt = ckpt_dir / f"step_{step:07d}.pt"
        save_ckpt(final_ckpt, step=step, epoch=epoch, model=model, opt=opt, args=args, rms=rms)
        save_ckpt(ckpt_dir / "latest.pt", step=step, epoch=epoch, model=model, opt=opt, args=args, rms=rms)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[   # paths to preprocessed frames
        "/<path>/expert-shards",
        "/<path>/mixed-small-shards",
        "/<path>/mixed-large-shards",
    ])
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)

    # image / patching
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--n_latents", type=int, default=16)
    p.add_argument("--d_bottleneck", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--mae_p_min", type=float, default=0.0)
    p.add_argument("--mae_p_max", type=float, default=0.9)
    p.add_argument("--qk_norm", action="store_true")
    p.add_argument("--rope", action="store_true",
                   help="rotary temporal positions instead of the absolute sinusoidal time table")
    p.add_argument("--pos_offset_max", type=int, default=0,
                   help="sample temporal position start in [0,N] each step; 0 disables")
    p.add_argument("--attn_softcap", type=float, default=0.0)

    # Better performance without scale_pos_embeds (set to False)
    # Set to True by default for backwards compatibility. Details here: https://github.com/nicklashansen/dreamer4/pull/4
    p.add_argument("--scale_pos_embeds", action="store_true")

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip_enc", type=float, default=1.0)
    p.add_argument("--grad_clip_dec", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=0)

    # collapse tripwire (debug/z_spread is logged every log_every steps)
    p.add_argument("--collapse_check_after", type=int, default=1000)
    p.add_argument("--collapse_spread_min", type=float, default=1e-3)

    # lpips
    p.add_argument("--lpips_weight", type=float, default=0.2)
    p.add_argument("--lpips_frac", type=float, default=0.5)
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # logging / viz
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=500)
    p.add_argument("--viz_max_items", type=int, default=4)
    p.add_argument("--viz_max_T", type=int, default=8)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-tokenizer")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/tokenizer_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())

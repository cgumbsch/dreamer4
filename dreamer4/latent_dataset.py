# latent_dataset.py
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentWindowDataset(Dataset):
    """Windows of precomputed tokenizer latents.

    The tokenizer is frozen during dynamics training, so encoding the same frames every epoch
    recomputes a fixed function. This reads the latents instead. Layout, written by the builder:

      root/latents.f32   float32 (N, T, n_latents, d_bottleneck), C order
      root/actions.f32   float32 (N, T, action_dim)
      root/act_mask.f32  float32 (N, T, action_dim)
      root/emb_id.i64    int64   (N,)
      root/index.json    metadata, including the tokenizer identity below

    `tokenizer_sha` / `preprocessing_hash` are compared against the sidecar and a mismatch raises:
    latents belong to one tokenizer, and training on the wrong ones is otherwise silent.
    """

    def __init__(
        self,
        root: str,
        *,
        seq_len: int,
        action_dim: int = 16,
        tokenizer_sha: Optional[str] = None,
        preprocessing_hash: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.root = str(root)
        with open(os.path.join(self.root, "index.json"), "r") as f:
            self.meta = json.load(f)

        if int(self.meta["seq_len"]) != int(seq_len):
            raise ValueError(
                f"latent cache was built at seq_len={self.meta['seq_len']}, "
                f"trainer asks for {seq_len}"
            )
        if int(self.meta["action_dim"]) != int(action_dim):
            raise ValueError(
                f"latent cache action_dim={self.meta['action_dim']} != {action_dim}"
            )
        if tokenizer_sha is not None and self.meta["tokenizer_sha256"] != tokenizer_sha:
            raise ValueError(
                "latent cache was built with a different tokenizer: "
                f"{self.meta['tokenizer_sha256'][:12]} vs {tokenizer_sha[:12]}"
            )
        if preprocessing_hash is not None and self.meta.get("preprocessing_hash") not in (
            None, preprocessing_hash
        ):
            raise ValueError(
                f"latent cache preprocessing {self.meta.get('preprocessing_hash')} "
                f"!= {preprocessing_hash}"
            )

        self.N = int(self.meta["n_windows"])
        self.T = int(self.meta["seq_len"])
        self.L = int(self.meta["n_latents"])
        self.D = int(self.meta["d_bottleneck"])
        self.A = int(self.meta["action_dim"])
        self.tasks = list(self.meta["tasks"])

        if self.N <= 0:
            raise RuntimeError(f"empty latent cache at {self.root}")

        self._lat = None
        self._act = None
        self._mask = None
        self._emb = None

        if verbose:
            print(f"[LatentWindowDataset] {self.N} windows, T={self.T}, "
                  f"latents {self.L}x{self.D}, tasks={len(self.tasks)}, root={self.root}")

    def _open(self):
        # opened lazily so DataLoader workers each get their own handles
        if self._lat is None:
            self._lat = np.memmap(os.path.join(self.root, "latents.f32"), dtype=np.float32,
                                  mode="r", shape=(self.N, self.T, self.L, self.D))
            self._act = np.memmap(os.path.join(self.root, "actions.f32"), dtype=np.float32,
                                  mode="r", shape=(self.N, self.T, self.A))
            self._mask = np.memmap(os.path.join(self.root, "act_mask.f32"), dtype=np.float32,
                                   mode="r", shape=(self.N, self.T, self.A))
            self._emb = np.memmap(os.path.join(self.root, "emb_id.i64"), dtype=np.int64,
                                  mode="r", shape=(self.N,))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        self._open()
        i = int(idx)
        return {
            "z": torch.from_numpy(np.asarray(self._lat[i])),
            "act": torch.from_numpy(np.asarray(self._act[i])),
            "act_mask": torch.from_numpy(np.asarray(self._mask[i])),
            "emb_id": torch.tensor(int(self._emb[i]), dtype=torch.long),
        }


def collate_latent_batch(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

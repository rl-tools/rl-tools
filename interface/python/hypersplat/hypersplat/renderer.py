"""gsplat-backed scene loader and renderer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData

import gsplat


def load_calibration(ply_path: str | Path) -> Optional[np.ndarray]:
    """Look for a sidecar `{name}.json` next to `{name}.ply` and return a 4x4 drone-ENU -> scene transform.

    Schema (all fields optional):
        {
          "rotation_to_upright": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
          "origin":              [x, y, z]
        }

    `rotation_to_upright` is the matrix `R` such that `p_upright = R @ p_scene` (this is
    what ground-plane calibration tools produce). `origin` is the point in scene coords
    where the drone-world origin sits. Returns None if no sidecar is found.
    """
    sidecar = Path(ply_path).with_suffix(".json")
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text())
    R = np.eye(3, dtype=np.float64)
    if "rotation_to_upright" in data:
        R = np.asarray(data["rotation_to_upright"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T            # scene_transform takes drone-ENU -> scene = R^T (R takes scene -> upright)
    T[:3, 3] = origin
    return T


@dataclass
class Scene:
    """A loaded 3D Gaussian Splatting scene held on `device`."""

    means: torch.Tensor       # (N, 3)
    quats: torch.Tensor       # (N, 4) wxyz, normalized
    scales: torch.Tensor      # (N, 3) linear scale (already exp'd)
    opacities: torch.Tensor   # (N,)   in [0, 1] (already sigmoid'd)
    colors: torch.Tensor      # (N, K, 3) SH coeffs if sh_degree>0 else (N, 3) RGB
    sh_degree: Optional[int]  # None for RGB-only colors
    device: torch.device
    scene_transform: Optional[np.ndarray] = None  # 4x4 drone-ENU -> scene (from sidecar JSON)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @classmethod
    def synthetic(
        cls,
        num_gaussians: int = 2000,
        bounds: float = 8.0,
        seed: int = 0,
        device: str | torch.device = "cuda",
    ) -> "Scene":
        """A toy scene of randomly-placed colored isotropic Gaussians.

        Useful as a fallback when no pretrained .ply is available — lets us exercise
        the full render path end-to-end without depending on external assets.
        """
        rng = np.random.default_rng(seed)
        means = rng.uniform(-bounds, bounds, size=(num_gaussians, 3)).astype(np.float32)
        # Hollow out the center so the camera spawn point isn't inside a Gaussian.
        keep = np.linalg.norm(means, axis=-1) > 1.5
        means = means[keep]
        N = means.shape[0]
        # wxyz quaternions, identity rotation.
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0
        scales = rng.uniform(0.05, 0.25, size=(N, 3)).astype(np.float32)
        opacities = np.full((N,), 0.9, dtype=np.float32)
        colors = rng.uniform(0.0, 1.0, size=(N, 3)).astype(np.float32)

        device = torch.device(device)
        return cls(
            means=torch.from_numpy(means).to(device),
            quats=torch.from_numpy(quats).to(device),
            scales=torch.from_numpy(scales).to(device),
            opacities=torch.from_numpy(opacities).to(device),
            colors=torch.from_numpy(colors).to(device),
            sh_degree=None,
            device=device,
        )

    @classmethod
    def from_ply(cls, path: str | Path, device: str | torch.device = "cuda") -> "Scene":
        """Load a 3DGS .ply (INRIA / nerfstudio export format)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"scene .ply not found: {path}")

        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        names = v.dtype.names

        means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

        # Quaternions stored as rot_0..rot_3 in wxyz order, may be unnormalized.
        quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True).clip(min=1e-12)

        # Scales stored in log space.
        scales_log = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
        scales = np.exp(scales_log)

        # Opacities stored as logits.
        opa_logit = v["opacity"].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opa_logit))

        # Colors: f_dc_{0..2} are the DC SH coefficients; f_rest_* are higher-order SH coefs.
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)  # (N, 3)
        rest_names = sorted(
            (n for n in names if n.startswith("f_rest_")),
            key=lambda n: int(n.split("_")[-1]),
        )
        if rest_names:
            n_rest = len(rest_names)              # 3 * (K - 1) where K = (deg+1)^2
            assert n_rest % 3 == 0, f"f_rest_* count {n_rest} not divisible by 3"
            k_minus_1 = n_rest // 3
            k = k_minus_1 + 1
            sh_degree = int(round(np.sqrt(k))) - 1
            assert (sh_degree + 1) ** 2 == k, f"unexpected SH count {k}"
            f_rest = np.stack([v[n] for n in rest_names], axis=-1).astype(np.float32)
            # INRIA layout is (N, 3, K-1) channels-first then per-band; reshape accordingly.
            f_rest = f_rest.reshape(-1, 3, k_minus_1).transpose(0, 2, 1)  # (N, K-1, 3)
            colors = np.concatenate([f_dc[:, None, :], f_rest], axis=1)   # (N, K, 3)
        else:
            sh_degree = None
            # Map DC coefficient to RGB roughly: rgb = 0.5 + C0 * f_dc, C0 = 0.28209479
            colors = (0.5 + 0.28209479177387814 * f_dc).clip(0.0, 1.0)

        device = torch.device(device)
        return cls(
            means=torch.from_numpy(means).to(device),
            quats=torch.from_numpy(quats).to(device),
            scales=torch.from_numpy(scales).to(device),
            opacities=torch.from_numpy(opacities).to(device),
            colors=torch.from_numpy(colors).to(device),
            sh_degree=sh_degree,
            device=device,
            scene_transform=load_calibration(path),
        )


def make_intrinsics(
    image_size: tuple[int, int],
    fov_deg: float,
    override: Optional[dict] = None,
) -> np.ndarray:
    """Build a 3x3 K matrix from (H, W) + horizontal FOV, or copy override values."""
    H, W = image_size
    if override is not None:
        return np.array(
            [[override["fx"], 0.0, override["cx"]],
             [0.0, override["fy"], override["cy"]],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    fx = (W / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, W / 2.0],
         [0.0, fx, H / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class Renderer:
    """Thin wrapper around gsplat.rasterization for a single camera."""

    def __init__(self, scene: Scene, image_size: tuple[int, int], K: np.ndarray):
        self.scene = scene
        self.H, self.W = image_size
        self.K = torch.from_numpy(K.astype(np.float32)).to(scene.device)

    def render(self, viewmat: np.ndarray) -> np.ndarray:
        """viewmat: (4,4) world->camera (OpenGL/gsplat convention). Returns uint8 (H, W, 3)."""
        s = self.scene
        viewmat_t = torch.from_numpy(viewmat.astype(np.float32)).to(s.device).unsqueeze(0)  # (1,4,4)
        K = self.K.unsqueeze(0)  # (1,3,3)

        kwargs = dict(
            means=s.means,
            quats=s.quats,
            scales=s.scales,
            opacities=s.opacities,
            colors=s.colors,
            viewmats=viewmat_t,
            Ks=K,
            width=self.W,
            height=self.H,
            render_mode="RGB",
        )
        if s.sh_degree is not None:
            kwargs["sh_degree"] = s.sh_degree

        with torch.no_grad():
            render, _alpha, _info = gsplat.rasterization(**kwargs)
        # render: (1, H, W, 3) float in [0,1]
        img = render[0].clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
        return img

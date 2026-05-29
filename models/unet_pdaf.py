from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBackbone(nn.Module):
    """Plain U-Net used by both teacher and student.

    We keep the encoder/decoder split explicit because PDAF needs:
    1. multi-scale encoder features for LPE/DPE conditioning
    2. a decoder that can consume DCM-modulated features
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _align_and_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def encode(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features: OrderedDict[str, torch.Tensor] = OrderedDict()
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        features["layer1"] = e1
        features["layer2"] = e2
        features["layer3"] = e3
        features["layer4"] = e4
        return features

    def decode(self, features: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        layer1 = features["layer1"]
        layer2 = features["layer2"]
        layer3 = features["layer3"]
        layer4 = features["layer4"]

        bottleneck = self.bottleneck(F.max_pool2d(layer4, 2))
        d4 = self.dec4(self._align_and_concat(self.up4(bottleneck), layer4))
        d3 = self.dec3(self._align_and_concat(self.up3(d4), layer3))
        d2 = self.dec2(self._align_and_concat(self.up2(d3), layer2))
        d1 = self.dec1(self._align_and_concat(self.up1(d2), layer1))
        return self.head(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def prepare_condition(
    features: OrderedDict[str, torch.Tensor],
    layer_names: list[str],
    reference_layer: str,
) -> torch.Tensor:
    """Collect selected feature maps into one conditioning tensor.

    `condition_layers` can be a subset of encoder stages. We first resize every
    chosen feature map to the spatial size of `reference_layer`, then
    concatenate them along the channel dimension.
    """
    h, w = features[reference_layer].shape[-2:]
    conds = []
    for name in layer_names:
        cond = features[name]
        if cond.shape[-2:] != (h, w):
            cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=False)
        conds.append(cond)
    return torch.cat(conds, dim=1)


class ResidualStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int) -> None:
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                ]
            )
        self.blocks = nn.ModuleList(
            [nn.Sequential(*blocks[i : i + 5]) for i in range(0, len(blocks), 5)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = F.leaky_relu(x + block(x), negative_slope=0.1, inplace=True)
        return x


class LatentPriorExtractor(nn.Module):
    """LPE in the paper.

    Input:
    - source_condition: condition tensor from the frozen teacher on source image
    - pseudo_target_condition: condition tensor from the frozen teacher on
      pseudo-target image

    Output:
    - z: sampled latent domain prior z_tilde
    - mu/logvar: variational parameters for the latent prior

    Note:
    `in_channels` here means the channels of one condition tensor. The actual
    first convolution expects `source || pseudo_target`, so it uses
    `in_channels * 2`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        prior_channels: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = ResidualStack(hidden_channels, num_blocks)
        self.mu = nn.Conv2d(hidden_channels, prior_channels, kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(hidden_channels, prior_channels, kernel_size=3, padding=1)

    def forward(
        self,
        source_condition: torch.Tensor,
        pseudo_target_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([source_condition, pseudo_target_condition], dim=1)
        hidden = self.body(self.stem(x))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class SFTBlock(nn.Module):
    """Convert latent prior into affine modulation parameters gamma/beta."""
    def __init__(self, prior_channels: int, feature_channels: int, modulation_scale: float = 1.0) -> None:
        super().__init__()
        self.modulation_scale = modulation_scale
        self.scale = nn.Sequential(
            nn.Conv2d(prior_channels, prior_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(prior_channels, feature_channels, kernel_size=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(prior_channels, prior_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(prior_channels, feature_channels, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if prior.shape[-2:] != feature.shape[-2:]:
            prior = F.interpolate(prior, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        gamma = self.scale(prior) * self.modulation_scale
        beta = self.shift(prior) * self.modulation_scale
        return feature * (1.0 + gamma) + beta


class DomainCompensationModule(nn.Module):
    """DCM in the paper.

    A single latent prior is broadcast back to several encoder stages so the
    student can compensate domain shift before decoding.
    """
    def __init__(self, prior_channels: int, feature_dims: dict[str, int], insert_layers: list[str], modulation_scale: float = 1.0) -> None:
        super().__init__()
        self.insert_layers = insert_layers
        self.blocks = nn.ModuleDict(
            {name: SFTBlock(prior_channels, feature_dims[name], modulation_scale=modulation_scale) for name in insert_layers}
        )

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        prior: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        output: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, feature in features.items():
            if name in self.blocks:
                output[name] = self.blocks[name](feature, prior)
            else:
                output[name] = feature
        return output


class Denoiser(nn.Module):
    """Small denoising network used inside DPE."""
    def __init__(self, latent_channels: int, cond_channels: int, hidden_channels: int, num_blocks: int, timesteps: int) -> None:
        super().__init__()
        self.time_scale = float(max(1, timesteps - 1))
        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels + cond_channels + 1, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body = ResidualStack(hidden_channels, num_blocks)
        self.out = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor, cond: torch.Tensor, step: int) -> torch.Tensor:
        b, _, h, w = latent.shape
        if cond.shape[-2:] != (h, w):
            cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=False)
        t = torch.full((b, 1, h, w), float(step) / self.time_scale, device=latent.device, dtype=latent.dtype)
        hidden = self.stem(torch.cat([latent, cond, t], dim=1))
        hidden = self.body(hidden)
        return self.out(hidden)


class DiffusionPriorEstimator(nn.Module):
    """DPE in the paper.

    Training:
    - receives the student's current condition tensor
    - denoises a noisy version of z_tilde and predicts z_hat

    Inference:
    - starts from pure Gaussian noise
    - predicts z_hat using only target-image features
    """
    def __init__(
        self,
        prior_channels: int,
        cond_channels: int,
        hidden_channels: int,
        num_blocks: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        self.prior_channels = prior_channels
        self.timesteps = timesteps
        self.denoiser = Denoiser(prior_channels, cond_channels, hidden_channels, num_blocks, timesteps)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, z0: torch.Tensor, step: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bars[step]
        return alpha_bar.sqrt() * z0 + (1.0 - alpha_bar).sqrt() * noise

    def sample(self, cond: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(
                cond.size(0),
                self.prior_channels,
                cond.size(2),
                cond.size(3),
                device=cond.device,
                dtype=cond.dtype,
            )
        current = latent
        for step in range(self.timesteps - 1, -1, -1):
            pred_noise = self.denoiser(current, cond, step)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            x0 = (current - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp(min=1e-6)
            if step > 0:
                alpha_bar_prev = self.alpha_bars[step - 1]
                current = alpha_bar_prev.sqrt() * x0 + (1.0 - alpha_bar_prev).sqrt() * pred_noise
            else:
                current = x0
        return current

    def forward(self, cond: torch.Tensor, target_prior: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if target_prior is None:
            noisy = torch.randn(
                cond.size(0),
                self.prior_channels,
                cond.size(2),
                cond.size(3),
                device=cond.device,
                dtype=cond.dtype,
            )
        else:
            noisy = self.q_sample(target_prior, self.timesteps - 1)
        return self.sample(cond, noisy), noisy


class PDAFUNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.condition_layers = model_cfg["condition_layers"]
        self.condition_reference = model_cfg["condition_reference"]
        self.insert_layers = model_cfg["insert_layers"]
        feature_dims = {
            "layer1": model_cfg["base_channels"],
            "layer2": model_cfg["base_channels"] * 2,
            "layer3": model_cfg["base_channels"] * 4,
            "layer4": model_cfg["base_channels"] * 8,
        }
        cond_channels = sum(feature_dims[name] for name in self.condition_layers)

        # Teacher corresponds to the paper's frozen pretrained segmentation
        # model (E_vartheta / D_vartheta). Student is the trainable model
        # adapted with PDAF (E_theta / D_theta).
        self.teacher = UNetBackbone(
            in_channels=model_cfg["in_channels"],
            base_channels=model_cfg["base_channels"],
            out_channels=model_cfg["out_channels"],
        )
        self.student = deepcopy(self.teacher)
        self.lpe = LatentPriorExtractor(
            in_channels=cond_channels,
            hidden_channels=model_cfg["lpe_hidden_channels"],
            prior_channels=model_cfg["prior_channels"],
            num_blocks=model_cfg["lpe_blocks"],
        )
        self.dcm = DomainCompensationModule(
            prior_channels=model_cfg["prior_channels"],
            feature_dims=feature_dims,
            insert_layers=self.insert_layers,
            modulation_scale=model_cfg.get("dcm_modulation_scale", 1.0),
        )
        self.dpe = DiffusionPriorEstimator(
            prior_channels=model_cfg["prior_channels"],
            cond_channels=cond_channels,
            hidden_channels=model_cfg["dpe_hidden_channels"],
            num_blocks=model_cfg["dpe_blocks"],
            timesteps=model_cfg["timesteps"],
            beta_start=model_cfg["beta_start"],
            beta_end=model_cfg["beta_end"],
        )

    def initialize_student_from_teacher(self) -> None:
        """Start PDAF training from the teacher weights, not from scratch."""
        self.student.load_state_dict(self.teacher.state_dict())

    def freeze_teacher(self) -> None:
        """Teacher only provides stable features/predictions during PDAF training."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def load_teacher(self, checkpoint: dict | str) -> None:
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.teacher.load_state_dict(state)
        self.initialize_student_from_teacher()
        self.freeze_teacher()

    def teacher_predict(self, x: torch.Tensor) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor]:
        """Return frozen teacher features and logits for one image batch."""
        features = self.teacher.encode(x)
        logits = self.teacher.decode(features)
        return features, logits

    def student_predict_with_prior(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor]:
        """Run student with one latent prior injected by DCM."""
        features = self.student.encode(x)
        fused = self.dcm(features, prior)
        logits = self.student.decode(fused)
        return features, logits

    def forward_train(
        self,
        source_x: torch.Tensor,
        pseudo_target_x: torch.Tensor,
        use_dpe: bool = True,
        use_gaussian: bool = False,
    ) -> dict[str, torch.Tensor]:
        """One PDAF training step.

        Flow:
        1. frozen teacher extracts source/pseudo-target features
        2. LPE builds the "optimal" latent prior z_tilde from that pair
        3. student + DCM(z_tilde) predicts logits_tilde
        4. DPE predicts deployable prior z_hat from student features
           - if use_gaussian=True: start from pure Gaussian noise (inference mode)
           - else: start from z_tilde + noise (conditional denoising mode)
        5. student + DCM(z_hat) predicts logits_hat

        DPE conditions on student(pseudo_target) features. This simulates inference
        where only target image is available - both training and inference use
        student features from the "target-like" image.
        """
        with torch.no_grad():
            teacher_source_features, teacher_source_logits = self.teacher_predict(source_x)
            teacher_target_features, _ = self.teacher_predict(pseudo_target_x)

        # LPE uses the teacher's paired source/pseudo-target features to learn
        # a structured latent prior for the current domain shift.
        source_cond = prepare_condition(
            teacher_source_features,
            self.condition_layers,
            self.condition_reference,
        )
        pseudo_target_cond = prepare_condition(
            teacher_target_features,
            self.condition_layers,
            self.condition_reference,
        )
        z_tilde, mu, logvar = self.lpe(source_cond, pseudo_target_cond)

        # First student branch: use the "ideal" prior z_tilde from LPE.
        _, logits_tilde = self.student_predict_with_prior(pseudo_target_x, z_tilde)

        # student_cond: use raw student features (BEFORE DCM modulation).
        # This ensures DPE learns from the same feature distribution it will
        # see at inference time.
        raw_student_features = self.student.encode(pseudo_target_x)
        student_cond = prepare_condition(
            raw_student_features,
            self.condition_layers,
            self.condition_reference,
        )

        # Second student branch: use the deployable prior z_hat estimated by DPE
        # from student features only, which mirrors inference-time behavior.
        if use_dpe:
            # Mode 1 (conditional denoising): target_prior=z_tilde → add noise → denoise
            # Mode 2 (pure Gaussian):         target_prior=None → pure noise → denoise
            target_prior = None if use_gaussian else z_tilde
            z_hat, z_noisy = self.dpe(student_cond, target_prior=target_prior)
            _, logits_hat = self.student_predict_with_prior(pseudo_target_x, z_hat)
        else:
            z_hat = z_tilde.detach()
            z_noisy = torch.zeros_like(z_tilde)
            logits_hat = logits_tilde.detach()

        return {
            "teacher_source_logits": teacher_source_logits,
            "teacher_source_features": teacher_source_features,
            "teacher_target_features": teacher_target_features,
            "logits_tilde": logits_tilde,
            "logits_hat": logits_hat,
            "z_tilde": z_tilde,
            "z_hat": z_hat,
            "z_noisy": z_noisy,
            "mu": mu,
            "logvar": logvar,
            "student_cond": student_cond,
            "raw_student_features": raw_student_features,
        }

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: student + DPE + DCM only, no teacher and no LPE."""
        features = self.student.encode(x)
        cond = prepare_condition(features, self.condition_layers, self.condition_reference)
        prior, _ = self.dpe(cond, target_prior=None)
        fused = self.dcm(features, prior)
        return self.student.decode(fused)

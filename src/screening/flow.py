import torch

# x: real image
# eps: noise
# z: noisy image = t * x + (1 - t) * eps
# t = 0 -> noise, t = 1 -> image
# v: velocity = image - noise


def image_pred_to_velocity_pred(
    pred_image: torch.Tensor,  # x [B, C, H, W]
    noisy_image: torch.Tensor,  # z [B, C, H, W]
    timestep: torch.Tensor,  # t [B]
    eps: float = 1e-5,
) -> torch.Tensor:
    return (pred_image - noisy_image) / (1 - timestep[:, None, None, None]).clamp(
        min=eps
    )

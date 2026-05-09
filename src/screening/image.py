from PIL import Image
import torch


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """
    (-infinity, infinity) -> [-1, 1] -> [0, 1] -> [0, 255] -> uint8 -> PIL Image
    """

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor * 255.0
    tensor = tensor.to(torch.uint8)
    array = tensor.cpu().numpy()
    array = array.transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    images: list[Image.Image] = []
    for i in range(array.shape[0]):
        images.append(Image.fromarray(array[i]))

    return images

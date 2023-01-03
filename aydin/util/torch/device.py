import torch


def get_torch_device():
    """
    Util function to get current torch device.

    Returns
    -------
    torch.device

    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"device {device}")
    return device

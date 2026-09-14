import random

import torch
import torchvision


class ScaleVideo(torch.nn.Module):
    def forward(self, video):
        return video.float().div(255.0)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, video):
        masked = video.clone()
        length = masked.size(0)

        mask_count = int((length + self.stride - 0.1) // self.stride)
        widths = torch.randint(0, self.window, size=(mask_count,))

        for width in widths:
            width = int(width)

            if width <= 0 or length - width <= 0:
                continue

            start = random.randint(0, length - width)
            masked[start : start + width] = 0

        return masked


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                ScaleVideo(),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset in ("val", "test"):
            self.video_pipeline = torch.nn.Sequential(
                ScaleVideo(),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        else:
            raise ValueError("subset must be train, val, or test.")

    def __call__(self, videos):
        return self.video_pipeline(videos)

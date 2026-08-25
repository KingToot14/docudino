import random

import torch
import torchvision.transforms.v2 as T

class ErodeDilateAugmentation(object):
    def __init__(self, kernel_size: int = 3, p: float = 0.2):
        """
        Randomly applies erosion or dilation to the input `image` with a kernel of `kernel_size`
        
        Args:
            image (torch.Tensor): The image to erode or dilate
            kernel_size (int): How large of a kernel to consider. Defaults to 3
            p (float): The odds of this augmentation activating. When activated, either erosion
                or dilation will occur with a 50/50 chance (mutually exclusive)
        """
        
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, image: torch.Tensor):
        # check if this should process
        if random.random() > self.p:
            return image

        kernel = torch.ones(
            1, 1, self.kernel_size, self.kernel_size,
            device=image.device,
            dtype=image.dtype,
        )
        
        # erode
        if random.random() < 0.50:
            x = torch.conv2d(1.0 - image, kernel, padding=self.kernel_size // 2)
            
            return ((1.0 - x) > 0).to(image.dtype)
        # dilate
        else:
            x = torch.conv2d(image, kernel, padding=self.kernel_size // 2)
            
            return (x > 0).to(image.dtype)

class TrainingAugmentations(object):
    def __init__(self, global_crop_scale: tuple[int, int],
                 local_crop_scale: tuple[int, int], n_local_crops: int,
                 augmentations: list[str] = ['erode_dilate']):
        """
        Args:
            global_crop_scale (tuple[int, int]): The minimum and maximum scale of global crops.
            local_crop_scale (tuple[int, int]): The minimum and maximum scale of local crops.
            n_local_cropss (int): The number of local crops to create.
            augmentations (list[str]): A list of data augmentations to apply. Can either be given
                in the format "<augmentation>" or "<augmentation>:<odds>".
        """
        
        # store parameters
        self.n_local_crops = n_local_crops
        
        # data augmentations
        data_augmentations = []
        
        for augmentation in augmentations:
            name: str = augmentation
            odds: float = None
            
            # parse name
            if ':' in name:
                name, odds = name.split(':')
                odds = float(odds)
            
            # add augmentations
            if augmentation == 'erode_dilate':
                if odds is not None:
                    data_augmentations.append(ErodeDilateAugmentation(p=odds))
                else:
                    data_augmentations.append(ErodeDilateAugmentation())
        
        data_augmentations = T.Compose(data_augmentations)
        
        # normalization
        to_image = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        
        normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        # define crops
        self.global_crop = T.Compose([
            to_image,
            T.RandomResizedCrop(224, scale=global_crop_scale, interpolation='bicubic'),
            data_augmentations,
            normalize,
        ])
        
        self.local_crop = T.Compose([
            to_image,
            T.RandomResizedCrop(96, scale=local_crop_scale, interpolation='bicubic'),
            data_augmentations,
            normalize,
        ])
        
        pass
    
    def __call__(self, image: torch.Tensor):
        crops: list[torch.Tensor] = []
        
        # global views
        crops.append(self.global_crop(image))
        crops.append(self.global_crop(image))
        
        # local views
        for _ in range(self.n_local_crops):
            crops.append(self.local_crop(image))
        
        return crops
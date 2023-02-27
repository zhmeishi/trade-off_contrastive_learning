from torchvision import transforms as T
from PIL import Image, ImageOps
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    torchvision.T.GaussianBlur = GaussianBlur


class BYOL_transform: # Table 6 
    def __init__(self, image_size, mean_std):
        self.single = False
        self.transform1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0)), # simclr paper gives the kernel size. Kernel size has to be odd positive number with torchvision
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.transform2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.RandomApply([GaussianBlur(kernel_size=int(0.1 * image_size))], p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
            T.RandomApply([Solarization()], p=0.2),
            
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])


    def __call__(self, x):
        x1 = self.transform1(x) 
        x2 = self.transform2(x) 
        return x1, x2


# class Transform_single:
#     def __init__(self, image_size, train, mean_std):
#         self.denormalize = Denormalize(*mean_std)
#         if train == True:
#             self.transform = T.Compose([
#                 T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
#                 T.RandomHorizontalFlip(),
#                 T.ToTensor(),
#                 T.Normalize(*mean_std)
#             ])
#         else:
#             self.transform = T.Compose([
#                 T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
#                 T.CenterCrop(image_size),
#                 T.ToTensor(),
#                 T.Normalize(*mean_std)
#             ])

#     def __call__(self, x):
#         return self.transform(x)



class Solarization():
    # ImageFilter
    def __init__(self, threshold=128):
        self.threshold = threshold
    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)



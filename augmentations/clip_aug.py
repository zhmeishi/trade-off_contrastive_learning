
import torchvision.transforms as T
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class CLIPTransform():
    def __init__(self, image_size, mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.single = False
        self.transform = T.Compose([
            _convert_image_to_rgb,
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 

class CLIPTransform_Single():
    def __init__(self, image_size, mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.single = True
        self.transform = T.Compose([
            _convert_image_to_rgb,
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        return self.transform(x)
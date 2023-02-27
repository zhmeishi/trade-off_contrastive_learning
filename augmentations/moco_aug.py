
import torchvision.transforms as T
try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    T.GaussianBlur = GaussianBlur
    

class MoCoTransform():
    def __init__(self, image_size, mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size 
        self.single = False
        self.transform = T.Compose([
            # T.RandomResizedCrop(image_size), # For CIFAR10 and MNIST
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 

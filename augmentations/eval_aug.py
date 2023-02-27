from torchvision import transforms as T
from PIL import Image


class Transform_single():
    def __init__(self, image_size, train, mean_std):
        self.single = True
        if train == True:
            self.transform = T.Compose([
                T.RandomResizedCrop((image_size, image_size), scale=(0.80, 1.0), ratio=(4.0/5.0,5.0/4.0), \
                    interpolation=T.InterpolationMode.BICUBIC),
                # T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                # T.RandomCrop(image_size, padding=4), # For CIFAR10 and MNIST
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])
        else:
            self.transform = T.Compose([
                # T.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256 
                # T.CenterCrop(image_size),
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])

    def __call__(self, x):
        return self.transform(x)

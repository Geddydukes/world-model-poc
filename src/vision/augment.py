import torchvision.transforms as T
def get_vision_transforms(image_size=224):
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

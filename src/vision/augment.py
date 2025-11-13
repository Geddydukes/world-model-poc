def get_vision_transforms(image_size=224):
    """Get vision transforms. Lazy imports torchvision to avoid heavy import cost."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

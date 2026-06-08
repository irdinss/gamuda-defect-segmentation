import albumentations as A


def get_train_transforms():

    return A.Compose([
        A.Resize(
            512,
            512,
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.RandomBrightnessContrast(
            p=0.3
        ),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),
    ])


def get_valid_transforms():

    return A.Compose([
        A.Resize(
            512,
            512,
        ),

        A.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),
    ])
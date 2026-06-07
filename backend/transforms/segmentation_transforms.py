import albumentations as A


def get_train_transforms():

    return A.Compose(
        [
            A.Resize(
                height=512,
                width=512
            ),

            A.HorizontalFlip(
                p=0.5
            ),

            A.VerticalFlip(
                p=0.5
            ),
        ]
    )


def get_valid_transforms():

    return A.Compose(
        [
            A.Resize(
                height=512,
                width=512
            )
        ]
    )
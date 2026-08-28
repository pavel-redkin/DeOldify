from fastai.vision.all import *
from fastai.data.core import DataLoaders
from pathlib import Path


def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=None,
) -> DataLoaders:

    if xtra_tfms is None:
        xtra_tfms = []

    src = Datasets(
        get_image_files(crappy_path),
        tfms=[[PILImage.create], [PILImage.create]],
        n_inp=1,
    )

    dls = src.dataloaders(
        bs=bs,
        after_item=[Resize(sz), Normalize.from_stats(*stats)],
        after_batch=[*aug_transforms(
            max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
        )],
        num_workers=num_workers,
        pin_memory=True,
    )

    dls.c = 3
    return dls


def get_dummy_databunch() -> DataLoaders:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )

from fastai.learner import Learner
from fastai.torch_core import *
from fastai.vision.all import *
from fastai.layers import NormType
from fastai.vision.learner import create_body
import torchvision.models as models
from torch import nn
from .unet import DynamicUnetWide, DynamicUnetDeep
from .dataset import get_colorize_data, get_dummy_databunch
from pathlib import Path


def _instantiate_arch(arch, pretrained: bool = True) -> nn.Module:
    """Instantiate an arch callable into an nn.Module, handling API changes."""
    if isinstance(arch, nn.Module):
        return arch
    try:
        return arch(pretrained=pretrained)
    except TypeError:
        pass
    try:
        return arch(weights='IMAGENET1K_V1' if pretrained else None)
    except TypeError:
        pass
    return arch()


def _safe_create_body(arch, pretrained: bool = True):
    """Create body from arch, working around fastai version incompatibilities."""
    try:
        body = create_body(arch, pretrained=pretrained)
        _ = list(body.children())
        return body
    except (AttributeError, TypeError):
        model = _instantiate_arch(arch, pretrained=pretrained)
        body = nn.Sequential(*list(model.children())[:-1])
        return body


# Weights are implicitly read from ./models/ folder
def gen_inference_wide(
    root_folder: Path, weights_name: str, nf_factor: int = 2, arch=models.resnet101) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_wide(
        data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_wide(
    data, gen_loss, arch=models.resnet101, nf_factor: int = 2
) -> Learner:
    return unet_learner_wide(
        data,
        arch=arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


def unet_learner_wide(
    data,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    blur: bool = False,
    self_attention: bool = False,
    y_range=None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: int = 1,
    **kwargs
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    body = _safe_create_body(arch, pretrained=pretrained)
    model = to_device(
        DynamicUnetWide(
            body,
            n_classes=3,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = Learner(data, model, **kwargs)
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


# Weights are implicitly read from ./models/ folder
def gen_inference_deep(
    root_folder: Path, weights_name: str, arch=models.resnet34, nf_factor: float = 1.5) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_deep(
        data=data, gen_loss=F.l1_loss, arch=arch, nf_factor=nf_factor
    )
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_deep(
    data, gen_loss, arch=models.resnet34, nf_factor: float = 1.5
) -> Learner:
    return unet_learner_deep(
        data,
        arch,
        wd=1e-3,
        blur=True,
        norm_type=NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


def unet_learner_deep(
    data,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    blur: bool = False,
    self_attention: bool = False,
    y_range=None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: float = 1.5,
    **kwargs
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    body = _safe_create_body(arch, pretrained=pretrained)
    model = to_device(
        DynamicUnetDeep(
            body,
            n_classes=3,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = Learner(data, model, **kwargs)
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

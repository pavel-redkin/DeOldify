import re

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


def _remap_legacy_state_dict(state_dict):
    """Remap state_dict keys from the legacy DeOldify/fastai module structure
    to the current structure.  Three kinds of renamed paths are handled:

    1. SelfAttention: old stores conv weights directly on .query/.key/.value;
       new wraps them in ConvLayer (nn.Sequential) so a .0 is inserted.
    2. res_block: old stores the block as self.conv → nn.Sequential;
       new returns nn.Sequential directly (named attr → numeric index).
    3. Middle conv block: old wraps in a module with self.layers;
       new is bare nn.Sequential.
    """
    new_sd = {}
    for k, v in state_dict.items():
        # SelfAttention: .query. → .query.0.  (ConvLayer is now Sequential)
        k = re.sub(r'\.(query|key|value)\.', r'.\1.0.', k)
        # res_block: old named .conv attr → numeric index .0.
        k = k.replace('.conv.', '.0.')
        # middle block: .N.layers. → .N.
        k = re.sub(r'(\d+)\.layers\.', r'\1.', k)
        new_sd[k] = v
    return new_sd


def _load_weights_with_remap(learn, weights_name):
    """Load weights with automatic key remapping for legacy checkpoints."""
    import torch
    from fastai.learner import load_model

    weights_path = learn.path / learn.model_dir / f'{weights_name}.pth'
    if not weights_path.exists():
        weights_path = learn.path / f'{weights_name}.pth'
    if not weights_path.exists():
        learn.load(weights_name)
        return

    sd = torch.load(weights_path, map_location='cpu', weights_only=False)
    sd = _remap_legacy_state_dict(sd)
    learn.model.load_state_dict(sd, strict=False)


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


def _safe_create_body(arch, pretrained: bool = True, cut=None):
    """Create body from arch, working around fastai version incompatibilities.

    fastai's create_body expects arch to be callable and instantiates it internally,
    but some installed versions don't do this correctly (passing the function through
    as-is). This wrapper detects that failure and manually instantiates + cuts.
    """
    try:
        body = create_body(arch, pretrained=pretrained, cut=cut)
        # Verify body is a real Module with children, not the raw function
        list(body.children())
        return body
    except (AttributeError, TypeError):
        pass

    model = _instantiate_arch(arch, pretrained=pretrained)
    cut = ifnone(cut, -2)
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    raise ValueError("cut must be either int or callable")


# Weights are implicitly read from ./models/ folder
def gen_inference_wide(
    root_folder: Path, weights_name: str, nf_factor: int = 2, arch=models.resnet101) -> Learner:
    data = get_dummy_databunch()
    learn = gen_learner_wide(
        data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.path = root_folder
    _load_weights_with_remap(learn, weights_name)
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
        default_device(),
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
    _load_weights_with_remap(learn, weights_name)
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
        default_device(),
    )
    learn = Learner(data, model, **kwargs)
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

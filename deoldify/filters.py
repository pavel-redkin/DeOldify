from fastai.learner import Learner
from fastai.torch_core import *
from fastai.vision.all import *
import cv2
from PIL import Image as PilImage
from deoldify import device as device_settings
import logging
import numpy as np
import PIL.Image
import torch


def normalize_funcs(mean, std):
    mean, std = tensor(mean).view(1, 3, 1, 1), tensor(std).view(1, 3, 1, 1)
    def norm(inp, do_x=True):
        x, y = inp
        if do_x:
            x = (x - mean.to(x.device)) / std.to(x.device)
        return x, y
    def denorm(x, do_x=True):
        if do_x:
            x = x * std.to(x.device) + mean.to(x.device)
        return x
    return norm, denorm


class IFilter:
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass


class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        
        if not device_settings.is_gpu():
            self.learn.model = self.learn.model.cpu()
        
        self.device = next(self.learn.model.parameters()).device
        self.norm, self.denorm = normalize_funcs(*stats)

    def _transform(self, image: PilImage) -> PilImage:
        return image

    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)

    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = PILImage.create(model_image)
        tensor = ToTensor()(x).to(self.device).unsqueeze(0).float()
        tensor.div_(255)
        tensor, y = self.norm((tensor, tensor), do_x=True)
        
        try:
            with torch.no_grad():
                out = self.learn.model(tensor)
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            logging.warning('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image

        if isinstance(out, (list, tuple)):
            out = out[0]
        out = self.denorm(out, do_x=False)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        out = np.clip(out, 0, 255).astype(np.uint8)
        return PilImage.fromarray(out)

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn, stats=stats)
        self.render_base = 16

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
        final = PilImage.fromarray(final)
        return final


class MasterFilter:
    def __init__(self, filters, render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image

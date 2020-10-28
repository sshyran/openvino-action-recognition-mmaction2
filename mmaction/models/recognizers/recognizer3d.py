from .base import BaseRecognizer
from ..registry import RECOGNIZERS


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def reshape_input(self, imgs, masks=None):
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if masks is not None:
            masks = masks.reshape((-1,) + masks.shape[2:])

        return imgs, masks, []

    def reshape_input_inference(self, imgs, masks=None):
        return imgs, masks, []

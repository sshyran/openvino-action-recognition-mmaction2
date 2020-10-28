from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def reshape_input(self, imgs, masks=None):
        batches = imgs.shape[0]
        num_segs = imgs.shape[0] // batches

        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        if masks is not None:
            masks = masks.reshape((-1,) + masks.shape[2:])

        return imgs, masks, [num_segs]

    def reshape_input_inference(self, imgs, masks=None):
        return imgs, masks, [1]

import timm  # noqa
import torchvision.models as models  # noqa

_BACKBONES = {
    "vit_large_patch14_dinov2meh": 'timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False, img_size=224)',
}


def load(name):
    return eval(_BACKBONES[name])

from config.cfg_multimodal import cfg as settings_multimodal

# ---------------------------------------------------------------------------------
# from lib.model.model_template import build_model as build_dev
from lib.model.model_multimodal import build_model as build_multimodal

# ---------------------------------------------------------------------------------
from lib.dataset import lmdb_patch_build_fn, lmdb_patch_collate_fn

# ---------------------------------------------------------------------------------
from lib.tracker import MultiModalMTracker

exp_register = dict()

# #################################################################################

exp_register.update({
    'multimodal': {
        'args': settings_multimodal,
        'model_builder': build_multimodal,
        'dataset_fn': [lmdb_patch_build_fn, lmdb_patch_collate_fn],
        'tracker': MultiModalMTracker,
    }
})

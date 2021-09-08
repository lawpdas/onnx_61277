import cv2
import os
import torch
import numpy as np
import random


class Tracker(object):
    def __init__(self):
        super(Tracker, self).__init__()

        # updated hyper-params
        self.vis = False

        self.template_sf = None
        self.template_sz = None

        self.search_sf = None
        self.search_sz = None

        # ---------------
        self.model = None

        self.template_feat_sz = None
        self.search_feat_sz = None

        self.template_info = None

        self.language = None  # (B, 768)
        self.init_box = None  # [x y x y]
        self.last_box = None
        self.last_pos = None
        self.last_size = None
        self.last_score = None
        self.last_image = None

        self.imw = None
        self.imh = None
        self.channel_average = None

        self.idx = 0

    def init(self, *args, **kwargs):
        assert NotImplemented

    def track(self, *args, **kwargs):
        assert NotImplemented

    def update_state(self, pred_box, pred_score, scale_f):  # [cx cy w h]
        sf_w, sf_h = scale_f

        if pred_score is None:
            self.last_score = 1
        else:
            self.last_score = pred_score

        delta_pos = (pred_box[:2] - 0.5) * self.search_sz / np.array([sf_w, sf_h])
        current_size = pred_box[2:] * self.search_sz / np.array([sf_w, sf_h])

        self.last_pos = self.last_pos + delta_pos
        self.last_size = current_size

        self.last_box = np.array([self.last_pos[0] - self.last_size[0] / 2,
                                  self.last_pos[1] - self.last_size[1] / 2,
                                  self.last_pos[0] + self.last_size[0] / 2,
                                  self.last_pos[1] + self.last_size[1] / 2])

        out_box = np.array(self.last_box)
        out_box = self.clip_box(out_box, margin=10)

        return out_box, self.last_score  # [x y w h]

    def update_hyper_params(self, hp: dict):
        if hp is not None:
            for key, value in hp.items():
                setattr(self, key, value)

    @staticmethod
    def crop_patch(im, box, scale_factor, out_size):  # [x, y, x, y]
        pos = (box[:2] + box[2:]) / 2
        wh = box[2:] - box[:2] + 1

        w_z = wh[0] + (scale_factor - 1) * np.mean(wh)
        h_z = wh[1] + (scale_factor - 1) * np.mean(wh)
        crop_sz = np.ceil(np.sqrt(w_z * h_z))

        x1 = pos[0] - crop_sz / 2
        y1 = pos[1] - crop_sz / 2

        a = out_size / crop_sz
        b = out_size / crop_sz
        c = -a * x1
        d = -b * y1

        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        patch = cv2.warpAffine(im, mapping,
                               (out_size, out_size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.mean(im, axis=(0, 1)))

        x, y, w, h = box
        out_box = np.array([x, y, x+w-1, y+h-1])

        out_box[0::2] = out_box[0::2] * a + c
        out_box[1::2] = out_box[1::2] * b + d

        out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
        out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)  # [x, y, x, y]

        return patch, out_box, [a, b]

    @staticmethod
    def to_pytorch(x):
        x = torch.Tensor(x.transpose(2, 0, 1)).cuda().unsqueeze(0)
        return x

    def clip_box(self, box, margin=0):
        imh = self.imh
        imw = self.imw

        x1, y1, x2, y2 = box

        x1 = min(max(0, x1), imw - margin)
        x2 = min(max(margin, x2), imw)
        y1 = min(max(0, y1), imh - margin)
        y2 = min(max(margin, y2), imh)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return np.array([x1, y1, w, h])

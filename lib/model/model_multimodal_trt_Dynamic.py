import time

import cv2
import importlib
from functools import partial
from typing import List, Tuple

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import torch
from torch import nn, Tensor


class MultimodalModel(nn.Module):

    def __init__(self, args):
        super(MultimodalModel, self).__init__()

        self.pretrained_param = None

        self.register_buffer("pytorch_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))  # RGB
        self.register_buffer("pytorch_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))  # RGB

        self.cfg = args
        self.num_tasks = self.cfg.neck.num_tasks
        self.output_sz = self.cfg.head.output_size
        self.use_language = self.cfg.use_language

        # build backbone
        backbone_module = importlib.import_module('lib.model.backbones')
        self.backbone = getattr(backbone_module, self.cfg.backbone.type)(self.cfg.backbone)

        # build neck
        neck_module = importlib.import_module('lib.model.necks')
        self.cfg.neck.in_channels_list = [c for c in self.backbone.out_channels_list]
        self.neck = getattr(neck_module, self.cfg.neck.type)(self.cfg.neck)

        # # build multi-head
        # head_module = importlib.import_module('lib.model.heads')
        # self.heads = nn.ModuleList([
        #     getattr(head_module, tp + 'TRT')(self.cfg.head) for tp in self.cfg.head.type])

        # build criterion
        criteria_module = importlib.import_module('lib.model.criteria')
        self.criteria = [
            partial(getattr(criteria_module, tp),
                    alpha_giou=self.cfg.criterion.alpha_giou,
                    alpha_l1=self.cfg.criterion.alpha_l1,
                    alpha_conf=self.cfg.criterion.alpha_conf
                    ) for tp in self.cfg.criterion.type]

    def forward(self,
                template: Tensor,  # (N, C, H, W)
                search: Tensor,  # (N, C, H, W)
                bert: Tensor,  # (1, N, 768)
                ):
        device = next(self.parameters()).device

        img_t = template.to(device)
        img_s = search.to(device)
        lang = bert.to(device)

        ns, _, hs, ws = img_s.shape

        t_features: List[Tensor] = self.backbone(self._pytorch_norm(img_t))  # (N, C, H, W)
        s_features: List[Tensor] = self.backbone(self._pytorch_norm(img_s))  # (N, C, H, W)

        task_features: List[Tuple] = self.neck(template_features=t_features,
                                               search_features=s_features,
                                               lang=lang, lang_mask=None)

        # predictions = [self.heads[0](*task_features[0]),  # visual feature
        #                self.heads[0](*task_features[1])]  # cross modal feature
        #
        # # Aggregation
        # weights = torch.cat((predictions[0][1], predictions[1][1]), dim=1).softmax(dim=1)  # (N, 2)
        # weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, 2, 1, 1, 1)
        #
        # prediction = self.heads[0](task_features[0][0] * weights[:, 0] + task_features[1][0] * weights[:, 1],
        #                            task_features[0][1] * weights[:, 0] + task_features[1][1] * weights[:, 1])
        #
        # outputs_coord = box_convert(prediction[0], in_fmt='xyxy', out_fmt='cxcywh')
        #
        # return outputs_coord, prediction[1]

        return task_features[0], task_features[1]

    def _pytorch_norm(self, img):
        img = img.div(255.0)
        img = img.sub(self.pytorch_mean).div(self.pytorch_std)
        return img


def build_model(args):
    model = MultimodalModel(args)
    return model


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
                        [0, b, d]]).astype(float)

    patch = cv2.warpAffine(im, mapping,
                           (out_size, out_size),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=np.mean(im, axis=(0, 1)))

    x, y, w, h = box
    out_box = np.array([x, y, x + w - 1, y + h - 1])

    out_box[0::2] = out_box[0::2] * a + c
    out_box[1::2] = out_box[1::2] * b + d

    out_box[0::2] = np.clip(out_box[0::2], 0, out_size - 1)
    out_box[1::2] = np.clip(out_box[1::2], 0, out_size - 1)  # [x, y, x, y]

    return patch, out_box, [a, b]


if __name__ == '__main__':
    from config.cfg_multimodal import cfg as settings
    import matplotlib.pyplot as plt

    # t = np.random.rand(1, 3, 128, 128)
    # s = np.random.rand(1, 3, 256, 256)
    # b = np.random.rand(1, 110, 768)
    # model = MultimodalTRT()

    im1 = cv2.imread('./0001.jpg')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    box1 = np.array([23, 88, 23+66, 88+55])
    im1 = crop_patch(im1, box1, 2, 128)[0]
    im1 = im1.transpose(2, 0, 1)

    im2 = cv2.imread('./0128.jpg')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    box2 = np.array([55, 82, 55+35, 82+29]) + 10
    im2 = crop_patch(im2, box2, 4, 256)[0]
    im2 = im2.transpose(2, 0, 1)

    ONNX_PATH = 'multimodal_language_E500_dynamic_pytorch18.onnx'
    TRT_PATH = 'multimodal_language_E500_dynamic.trt'

    model = build_model(settings.model).cuda()
    # load_ckp(os.path.join(path_register.checkpoint_dir, 'multimodal_language/multimodal_language_E500.pth'), model)

    b = np.random.rand(1, 100, 768)
    t = torch.from_numpy(im1).float().cuda().unsqueeze(0)
    x = torch.from_numpy(im2).float().cuda().unsqueeze(0)
    bert = torch.from_numpy(b).float().cuda()

    tic = time.time()
    for _ in range(100):
        out_box, out_score = model(t, x, bert)
    print(out_box, out_score, 100 / (time.time() - tic))

    input_names = ["template"] + ["search"] + ["bert"]
    output_names = ["box"] + ["confidence"]

    # torch.onnx.export(
    #     model, (t, x, bert),
    #     ONNX_PATH,
    #     verbose=True,
    #     opset_version=11,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes={'bert': {1: 'seq_length'}}
    # )

    # ---------------------------
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # ---------------------------
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(ONNX_PATH, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print('Completed parsing of ONNX file')

        with builder.create_builder_config() as config:
            # This determines the amount of memory available to the builder when building an optimized engine
            # and should generally be set as high as possible.
            config.max_workspace_size = 1 << 30  # 2**30 Byte, 2*10 MB, 1 GB
            # config.flags = 1 << trt.BuilderFlag.FP16

            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(2).name, (1, 2, 768), (1, 50, 768), (1, 200, 768))
            config.add_optimization_profile(profile)

            print('Serialize the engine')
            with builder.build_serialized_network(network, config) as serialized_engine:
                # save
                print('Write the serialized engine to a file')
                with open(TRT_PATH, 'wb') as f:
                    f.write(serialized_engine)

                # directly use
                print('Deserialize engine to perform inference')
                with trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # ---------------------------
    print('Load serialized engine & Deserialize engine to perform inference')
    with open(TRT_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # Allocate some host and device buffers for inputs and outputs
    # Determine dimensions and create page-locked memory buffers
    # (i.e. won't be swapped to disk) to hold host inputs/outputs.
    inputs = []
    outputs = []
    bindings = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        if size < 0:  # set dynamic siz <<< !!!!!!!!!!!!
            size = abs(size) * 200

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    with engine.create_execution_context() as context:
        inputs[0]['host'] = im1.reshape(-1).astype(np.float32)
        inputs[1]['host'] = im2.reshape(-1).astype(np.float32)

        tic = time.time()
        for seq_len in range(100):
            inputs[2]['host'] = np.random.rand(1, seq_len+2, 768).reshape(-1).astype(np.float32)

            context.set_binding_shape(2, (1, seq_len+2, 768))  # set dynamic shape <<< !!!!!!!!!!!!

            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
            # cuda.memcpy_htod_async(d_input, h_input, stream)

            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
            # cuda.memcpy_dtoh_async(h_output, d_output, stream)

            # Synchronize the stream
            stream.synchronize()

            [print(output['host']) for output in outputs]
        print(100 / (time.time() - tic))

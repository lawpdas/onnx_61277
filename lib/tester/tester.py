import os
import cv2
import time
import numpy as np
import multiprocessing

import torch
# from got10k.trackers import Tracker as GOT10k_Tracker

from lib.register.benchmarks import benchmark_register as benchmark_loader
from lib.tracker import MultiModalMTracker


def load_ckp(ckp_path, model, nlp_model=None):
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    map_location = 'cpu'
    ckp = torch.load(ckp_path, map_location=map_location)

    model_dict = model.state_dict()
    ckp_dict = ckp['model']

    pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
    unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
    lost_param = [k for k, v in model_dict.items() if k not in ckp_dict]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('<Visual> load checkpoint from:', ckp_path)
    print('<Visual> unused param:', unused_param)
    print('<Visual> lost_param:', lost_param)

    if nlp_model is not None:
        model_dict = nlp_model.state_dict()
        if ckp.get('nlp_model', None) is None:
            raise Exception('use nlp model, but cannot find nlp_ckp')
        else:
            nlp_dict = ckp['nlp_model']

        pretrained_dict = {k: v for k, v in nlp_dict.items() if k in model_dict}
        unused_param = [k for k, v in nlp_dict.items() if k not in model_dict]
        lost_param = [k for k, v in model_dict.items() if k not in nlp_dict]

        nlp_dict.update(pretrained_dict)
        nlp_model.load_state_dict(model_dict)

        print('<NLP> load checkpoint from:', ckp_path)
        print('<NLP> unused param:', unused_param)
        print('<NLP> lost_param:', lost_param)
    else:
        pass


class Tester(object):
    def __init__(self, **kwargs):
        self.args = kwargs
        self.exp_cfg = kwargs.get('args', None)
        self.tester_cfg = kwargs.get('tester', None)
        self.tracker_cfg = kwargs.get('tracker', None)

        self.model = None
        self.nlp_model = None
        self.tracker = None

    def create_tracker(self):
        model_builder = self.args.get('model_builder', None)
        self.model = model_builder(self.exp_cfg.model).eval().cuda()

        if self.exp_cfg.model.use_language:
            from lib.model.nlp_models import BERT
            self.nlp_model = BERT(lr_mult=0).eval().cuda()

        if not os.path.exists(self.tracker_cfg.ckp_path):
            print('not find ckp path: {}'.format(self.tracker_cfg.ckp_path))
            raise AssertionError

        # self.load_ckp()
        load_ckp(self.tracker_cfg.ckp_path, self.model, nlp_model=self.nlp_model)

        self.tracker = MultiModalMTracker(hyper=self.tracker_cfg, model=self.model)

    def load_ckp(self):

        # map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        map_location = 'cpu'
        ckp = torch.load(self.tracker_cfg.ckp_path, map_location=map_location)

        model_dict = self.model.state_dict()
        ckp_dict = ckp['model']

        pretrained_dict = {k: v for k, v in ckp_dict.items() if k in model_dict}
        unused_param = [k for k, v in ckp_dict.items() if k not in model_dict]
        lost_param = [k for k, v in model_dict.items() if k not in ckp_dict]

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        print('<Visual> load checkpoint from:', self.tracker_cfg.ckp_path)
        print('<Visual> unused param:', unused_param)
        print('<Visual> lost_param:', lost_param)

        if self.exp_cfg.model.use_language:
            model_dict = self.nlp_model.state_dict()

            if ckp.get('nlp_model', None) is None:
                raise Exception('use nlp model, but cannot find nlp_ckp')
            else:
                nlp_dict = ckp['nlp_model']

            pretrained_dict = {k: v for k, v in nlp_dict.items() if k in model_dict}
            unused_param = [k for k, v in nlp_dict.items() if k not in model_dict]
            lost_param = [k for k, v in model_dict.items() if k not in nlp_dict]

            nlp_dict.update(pretrained_dict)
            self.nlp_model.load_state_dict(model_dict)

            print('<NLP> load checkpoint from:', self.tracker_cfg.ckp_path)
            print('<NLP> unused param:', unused_param)
            print('<NLP> lost_param:', lost_param)
        else:
            pass

    def save_common(self, video_name, box_list, time_list, score_list):
        res_path = os.path.join(self.tester_cfg.res_dir, '{}.txt'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, 'times', '{}_time.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'scores', '{}_confidence.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')

    def save_got10k(self, video_name, box_list, time_list, score_list):
        os.makedirs(os.path.join(self.tester_cfg.res_dir, video_name), exist_ok=True)

        res_path = os.path.join(self.tester_cfg.res_dir, video_name, '{}_001.txt'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, video_name, '{}_time.txt'.format(video_name))
        time2_path = os.path.join(self.tester_cfg.res_dir, 'times', '{}_time.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'scores', '{}_confidence.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(time2_path, time_list, fmt='%.8f', delimiter=',')
        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')

    def save_votlt(self, video_name, box_list, time_list, score_list):
        os.makedirs(os.path.join(self.tester_cfg.res_dir, 'longterm', video_name), exist_ok=True)

        res_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name, '{}_001.txt'.format(video_name))
        score_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name,
                                  '{}_001_confidence.value'.format(video_name))
        time_path = os.path.join(self.tester_cfg.res_dir, 'longterm', video_name, '{}_time.txt'.format(video_name))

        np.savetxt(res_path, box_list, fmt='%.3f', delimiter=',')
        with open(res_path, 'r') as f:
            ori = f.readlines()
            ori[0] = '1\n'
        with open(res_path, "w") as f:
            f.writelines(ori)

        np.savetxt(score_path, score_list, fmt='%.3f', delimiter=',')
        with open(score_path, 'r') as f:
            ori = f.readlines()
            ori[0] = '\n'
        with open(score_path, "w") as f:
            f.writelines(ori)

        np.savetxt(time_path, time_list, fmt='%.8f', delimiter=',')

    def run_seq(self, seq_info, num_gpu):
        video_idx, video_name, im_list, gt_list, lang, seq_num = seq_info

        try:
            worker_name = multiprocessing.current_process().name
            worker_id = int(worker_name.split('-')[1]) - 1
            gpu_id = worker_id % num_gpu
            torch.cuda.set_device(gpu_id)
            rank = worker_id
            print('start rank {}'.format(rank))
        except IndexError:
            rank = 0
            print('Not multi-processes !')
            torch.cuda.set_device(0)

        # skip some videos
        if ('lasot' in self.tester_cfg.benchmark or 'tnl2k' in self.tester_cfg.benchmark
                or 'trackingnet' in self.tester_cfg.benchmark or 'otb99lang' in self.tester_cfg.benchmark):
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, '{}.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return

        if 'got10k' in self.tester_cfg.benchmark:
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, video_name, '{}_001.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return

        if 'lt' in self.tester_cfg.benchmark:
            if os.path.exists(os.path.join(self.tester_cfg.res_dir, 'longterm', video_name, 
                                           '{}_001.txt'.format(video_name))):
                print('skip: ', self.tracker_cfg.name, '--', video_idx, video_name)
                return

        self.create_tracker()

        # word embedding
        if self.exp_cfg.model.use_language:
            lang_encode = self.nlp_model(lang)  # (N, L, 768)
        else:
            lang_encode = None

        fps_list = []

        box_list = np.zeros([len(im_list), 4])
        score_list = np.zeros([len(im_list), 1])
        time_list = np.zeros([len(im_list), 1])

        for i_im, im_f in enumerate(im_list):
            img = cv2.imread(im_f)
            gt = np.array(gt_list[0])  # [x y w h]

            tic = time.time()
            if i_im == 0:
                self.tracker.init(img, gt, language=lang_encode)  # [x y w h]
                predict_box, predict_score = np.array(gt), 1

                delta_time = time.time() - tic
                fps_list.append(1 / delta_time)

                time_list[i_im, :] = delta_time
                score_list[i_im, :] = predict_score
                box_list[i_im, :] = predict_box

            else:
                predict_box, predict_score = self.tracker.track(img)  # [x y w h]

                delta_time = time.time() - tic
                fps_list.append(1 / delta_time)

                time_list[i_im, :] = delta_time
                score_list[i_im, :] = predict_score
                box_list[i_im, :] = predict_box

        # Please update `func_results_load` if you add new benchmark
        if ('lasot' in self.tester_cfg.benchmark or 'tnl2k' in self.tester_cfg.benchmark
                or 'trackingnet' in self.tester_cfg.benchmark or 'otb99lang' in self.tester_cfg.benchmark):
            self.save_common(video_name, box_list, time_list, score_list)

        if 'got10k' in self.tester_cfg.benchmark:
            self.save_got10k(video_name, box_list, time_list, score_list)

        if 'lt' in self.tester_cfg.benchmark:
            self.save_votlt(video_name, box_list, time_list, score_list)

        print('[Rank {:2d}] {:3d}/{:d} {:<30s} -- [{:6.2f} fps]'.format(
            rank, video_idx, seq_num, video_name, np.mean(fps_list)))

    def run_ope(self):  # [x y x y]
        assert self.tester_cfg.num_gpu > 0, "need gpu for running"

        seqs = benchmark_loader[self.tester_cfg.benchmark]()

        if self.tester_cfg.num_process == 0:
            for video_idx, (video_name, im_list, gt_list, lang) in enumerate(seqs):

                seq_info = [video_idx, video_name, im_list, gt_list, lang, len(seqs)]
                self.run_seq(seq_info, self.tester_cfg.num_gpu)
        else:
            multiprocessing.set_start_method('spawn', force=True)
            print('>>> multi-processes running <<<')

            param_list = [([video_idx, video_name, im_list, gt_list, lang, len(seqs)],
                           self.tester_cfg.num_gpu)
                          for video_idx, (video_name, im_list, gt_list, lang) in enumerate(seqs)]
            
            with multiprocessing.Pool(processes=self.tester_cfg.num_process) as pool:
                pool.starmap(self.run_seq, param_list)

        # eval_fun = EVAL_FUN.get(benchmark_name, None)
        # if eval_fun is not None:
        #     eval_fun(self.save_name, self.tracker_name)

#
# class GOT10kRunner(GOT10k_Tracker):
#     def __init__(self, args: dict):
#         super(GOT10kRunner, self).__init__(
#             name=args['tracker_name'],  # tracker name
#             is_deterministic=True  # stochastic (False) or deterministic (True)
#         )
#
#         model = MODEL_ARCH[args['model_arch']](MODEL_CFG[args['model_arch']])
#
#         if not os.path.exists(self.args.get('ckp_path', '')):
#             print('not find: {}'.format(self.args['ckp_path']))
#             raise AssertionError
#         load_ckp(model=model, ckp_path=self.args['ckp_path'])
#
#         self.tracker = BaseTracker(model=model, hyper=args)
#
#     def init(self, image, box):  # [x y w h]
#         im = np.array(image)  # to BGR opencv style
#         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#
#         gt = np.array(box)
#
#         self.tracker.init(im, gt)
#
#     def update(self, image):
#         im = np.array(image)  # to BGR opencv style
#         im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
#
#         predict_box, predict_score = self.tracker.track(im)  # [x y w h]
#
#         return predict_box


# def vot_run(args: dict):
#     from trackers import vot
#
#     model = MODEL_ARCH[args['model_arch']](MODEL_CFG[args['model_arch']])
#
#     if not os.path.exists(args.get('ckp_path', '')):
#         print('not find: {}'.format(args['ckp_path']))
#         raise AssertionError
#     load_ckp(model=model, ckp_path=args['ckp_path'])
#
#     tracker = BaseTracker(model=model, hyper=args)
#
#     handle = vot.VOT("rectangle")
#     region = handle.region()
#
#     image_file = handle.frame()
#     if not image_file:
#         sys.exit(0)
#
#     image = cv2.imread(image_file)
#     gt = np.array([region.x, region.y, region.width, region.height])
#     tracker.init(image, gt)  # [x y w h]
#
#     while True:
#         image_file = handle.frame()
#         if not image_file:
#             break
#         image = cv2.imread(image_file)
#
#         region, confidence = tracker.track(image)  # [x y w h]
#         region = vot.Rectangle(region[0], region[1], region[2], region[3])  # [x y w h]
#         handle.report(region, confidence)
#
#
# def create_workspace(
#         workspace,
#         project_path,
#         tag,
#         args,
#         stack,
# ):
#     os.makedirs(workspace, exist_ok=True)
#
#     with open(os.path.join(workspace, 'config.yaml'), 'w') as f:
#         f.write('registry:\n')
#         f.write('- ./trackers.ini\n')
#         f.write('stack: {}\n'.format(stack))
#
#     with open(os.path.join(workspace, 'trackers.ini'), 'w') as f:
#         f.write('[{}]\n'.format(tag))
#         f.write('label = {}\n'.format(tag))
#         f.write('protocol = traxpython\n')
#         if args is None:
#             f.write('command = from trackers.runner import VOT_Run; VOT_Run()\n')
#         else:
#             f.write('command = from trackers.runner import VOT_Run; VOT_Run({})\n'.format(
#                 f"args={{ \'model_arch\': \'{args['model_arch']}\', \'final_epoch\': {args['final_epoch']:d} }}")
#             )
#         f.write('paths = {}\n'.format(os.path.join(project_path)))
#         f.write('env_PATH = %s;%s;${PATH}\n' % (project_path, os.path.join(project_path, 'trackers')))
#
#     if not os.path.exists(os.path.join(workspace, 'sequences')):
#         os.system('ln -s {} {}'.format(paths.eval_vot20, os.path.join(workspace, 'sequences')))


import os
import numpy as np
from tqdm import tqdm
from typing import List

strList = List[str]


def load_votlt(root):
    video_name_list = np.loadtxt(os.path.join(root, 'list.txt'), dtype=np.str).tolist()

    if len(video_name_list) > 35:
        name = 'VOT19-LT'
    else:
        name = 'VOT18-LT'

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading %s' % name):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n, 'color')
        lang_path = os.path.join(root + '_language', v_n, 'language.txt')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]
        with open(lang_path, 'r') as f:
            lang = f.readline()

        video_list.append([v_n, ims, gts, lang])

    return video_list


def load_lasot(root):
    video_name_list = np.loadtxt(os.path.join(root, '..', 'testing_set.txt'), dtype=np.str).tolist()

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading LaSOT'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        lang_path = os.path.join(root, v_n, 'nlp.txt')
        im_dir = os.path.join(root, v_n, 'img')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)

        gts = np.loadtxt(gt_path, delimiter=',').tolist()  # [x y w h]
        with open(lang_path, 'r') as f:
            lang = f.readline()
        ims = [os.path.join(im_dir, im_f) for im_f in im_list if 'jpg' in im_f]

        video_list.append([v_n, ims, gts, lang])

    return video_list


def load_got10k(root):
    video_name_list = os.listdir(root)
    video_name_list = sorted(video_name_list)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(root, v))]

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading GOT10k'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        im_dir = os.path.join(root, v_n)

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)
        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_tnl2k(root):
    video_name_list = os.listdir(root)
    video_name_list = sorted(video_name_list)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(root, v))]

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading TNL2K'):
        gt_path = os.path.join(root, v_n, 'groundtruth.txt')
        lang_path = os.path.join(root, v_n, 'language.txt')
        im_dir = os.path.join(root, v_n, 'imgs')

        im_list: strList = os.listdir(im_dir)
        im_list = sorted(im_list)
        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        # test_015_Sord_video_Q01_done/imgs/00000492.pn_error

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        with open(lang_path, 'r') as f:
            lang = f.readline()
            if len(lang) == 0:  # Fight_video_6-Done
                lang = '[MASK]'
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, lang])

    return video_list


def load_trackingnet(root):
    video_root = os.path.join(root, 'frames')
    annos_root = os.path.join(root, 'anno')

    video_name_list = os.listdir(video_root)
    video_name_list = sorted(video_name_list)
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading TrackingNet'):
        gt_path = os.path.join(annos_root, v_n + '.txt')
        im_dir = os.path.join(video_root, v_n)

        im_list = os.listdir(im_dir)
        im_list = sorted(im_list)

        ids = []
        for f in im_list:
            ids.append(int(f.split('.')[0]))
        im_list = np.array(im_list)
        im_list = im_list[np.argsort(np.array(ids))].tolist()

        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        # test_015_Sord_video_Q01_done/imgs/00000492.pn_error

        gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        video_list.append([v_n, ims, gts, ''])

    return video_list


def load_otb_lang(root):
    video_root = os.path.join(root, 'OTB_videos')
    annos_root = os.path.join(root, 'OTB_videos')
    lang_root = os.path.join(root, 'OTB_query_test')

    video_name_list = os.listdir(os.path.join(root, 'OTB_query_test'))
    video_name_list = sorted(video_name_list)
    video_name_list = [v.replace('.txt', '') for v in video_name_list if '.txt' in v]
    video_name_list = [v for v in video_name_list if os.path.isdir(os.path.join(video_root, v))]

    video_list = []
    for v_n in tqdm(video_name_list, ascii=True, desc='loading OTB-LANG'):
        gt_path = os.path.join(annos_root, v_n, 'groundtruth_rect.txt')
        im_dir = os.path.join(video_root, v_n, 'img')
        lang_path = os.path.join(lang_root, v_n + '.txt')

        im_list = os.listdir(im_dir)
        im_list = sorted(im_list)

        ids = []
        for f in im_list:
            ids.append(int(f.split('.')[0]))
        im_list = np.array(im_list)
        im_list = im_list[np.argsort(np.array(ids))].tolist()

        im_list = [im_f for im_f in im_list if 'jpg' in im_f or 'png' in im_f]
        ims = [os.path.join(im_dir, im_f) for im_f in im_list]

        try:
            gts = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4).tolist()  # [x y w h]
        except:
            gts = np.loadtxt(gt_path, delimiter='\t').reshape(-1, 4).tolist()  # [x y w h]

        with open(lang_path, 'r') as f:
            lang = f.readline().strip()

        video_list.append([v_n, ims, gts, lang])

    return video_list

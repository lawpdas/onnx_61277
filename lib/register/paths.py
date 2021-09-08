import sys
import os
import socket
from os.path import join as p_join
from easydict import EasyDict as Edict

path_register = Edict()

path_register.project_dir = '/home/space/Documents/Experiments/AAAI22_5381_code'
assert os.path.exists(path_register.project_dir), "Please set PATH in `lib/register/paths` !"

sys.path.append(path_register.project_dir)
sys.path.append(p_join(path_register.project_dir, 'lib'))


path_register.log_dir = p_join(path_register.project_dir, 'logs')
path_register.checkpoint_dir = p_join(path_register.project_dir, 'checkpoints')
path_register.result_dir = p_join(path_register.project_dir, 'results')
path_register.report_dir = p_join(path_register.project_dir, 'reports')
path_register.pretrain_dir = p_join(path_register.project_dir, 'pretrain')
path_register.vot_workspace_dir = p_join(path_register.project_dir, 'workspaces')


# ############################################
#              lmdb and json
# ############################################
if socket.gethostname() == 'z390f':
    lmdb_dir = '/data3/LMDB'
else:
    lmdb_dir = '/media/space/T7/LMDB'

path_register.lmdb = Edict()
path_register.json = Edict()

# got10k
path_register.lmdb.got10k_train = p_join(lmdb_dir, 'got10k_train')
path_register.lmdb.got10k_train_vot = p_join(lmdb_dir, 'got10k_train')
path_register.lmdb.got10k_val = p_join(lmdb_dir, 'got10k_val')

path_register.json.got10k_train = p_join(path_register.lmdb.got10k_train, 'got10k_train.json')
path_register.json.got10k_train_vot = p_join(p_join(lmdb_dir, 'got10k_train_vot'), 'got10k_train_vot.json')
path_register.json.got10k_val = p_join(path_register.lmdb.got10k_val, 'got10k_val.json')

# lasot
path_register.lmdb.lasot_train = p_join(lmdb_dir, 'lasot_train')
path_register.lmdb.lasot_val = p_join(lmdb_dir, 'lasot_val')

path_register.json.lasot_train = p_join(path_register.lmdb.lasot_train, 'lasot_train.json')
path_register.json.lasot_val = p_join(path_register.lmdb.lasot_val, 'lasot_val.json')

# trackingnet
path_register.lmdb.trackingnet_train_p0 = p_join(lmdb_dir, 'trackingnet_train_p0')
path_register.lmdb.trackingnet_train_p1 = p_join(lmdb_dir, 'trackingnet_train_p1')
path_register.lmdb.trackingnet_train_p2 = p_join(lmdb_dir, 'trackingnet_train_p2')

path_register.json.trackingnet_train_p0 = p_join(path_register.lmdb.trackingnet_train_p0, 'trackingnet_train_p0.json')
path_register.json.trackingnet_train_p1 = p_join(path_register.lmdb.trackingnet_train_p1, 'trackingnet_train_p1.json')
path_register.json.trackingnet_train_p2 = p_join(path_register.lmdb.trackingnet_train_p2, 'trackingnet_train_p2.json')

# tnl2k
path_register.lmdb.tnl2k_train = p_join(lmdb_dir, 'tnl2k_train')
path_register.lmdb.tnl2k_test = p_join(lmdb_dir, 'tnl2k_test')

path_register.json.tnl2k_train = p_join(path_register.lmdb.tnl2k_train, 'tnl2k_train.json')
path_register.json.tnl2k_test = p_join(path_register.lmdb.tnl2k_test, 'tnl2k_test.json')

# vid_sentence
path_register.lmdb.vid_sent_train = p_join(lmdb_dir, 'vid_sent_train')
path_register.lmdb.vid_sent_val = p_join(lmdb_dir, 'vid_sent_val')

path_register.json.vid_sent_train = p_join(path_register.lmdb.vid_sent_train, 'vid_sent_train.json')
path_register.json.vid_sent_val = p_join(path_register.lmdb.vid_sent_val, 'vid_sent_val.json')


# ############################################
#                benchmark
# ############################################
if socket.gethostname() == 'z390f':
    benchmark_dir = '/home/space/Documents/Datasets'
else:
    benchmark_dir = '/home/space/Documents/Datasets'

path_register.benchmark = Edict()

path_register.benchmark.got10k_test = p_join(benchmark_dir, 'GOT-10k/test')
path_register.benchmark.got10k_val = p_join(benchmark_dir, 'GOT-10k/val')
path_register.benchmark.trackingnet = p_join(benchmark_dir, 'TrackingNet/TEST')
path_register.benchmark.lasot = p_join(benchmark_dir, 'LaSOT/images')
path_register.benchmark.tnl2k = p_join(benchmark_dir, 'TNL2K/TNL2K_test_subset')
path_register.benchmark.vot20 = p_join(benchmark_dir, 'VOT2020')
path_register.benchmark.vot19lt = p_join(benchmark_dir, 'VOT2019-LT')
path_register.benchmark.vot18lt = p_join(benchmark_dir, 'VOT2018-LT')
path_register.benchmark.otb99lang = p_join(benchmark_dir, 'OTB99-LANG')
path_register.benchmark.vid_sentence = p_join(benchmark_dir, 'VID-Sentence')


if __name__ == '__main__':
    import pprint

    pp = pprint.PrettyPrinter(width=80, compact=False)
    pp.pprint(path_register)

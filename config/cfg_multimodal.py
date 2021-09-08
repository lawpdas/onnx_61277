from easydict import EasyDict as Edict

cfg = Edict()

# ############################################
#                  model
# ############################################
cfg.model = Edict()

# nlp
cfg.model.use_language = True
cfg.model.nlp_model = Edict()
cfg.model.nlp_model.type = 'BERT'
cfg.model.nlp_model.lr_mult = 0.0

# backbone
cfg.model.backbone = Edict()
cfg.model.backbone.type = 'Resnet'

cfg.model.backbone.arch = 'resnet50'
cfg.model.backbone.use_pretrain = True
cfg.model.backbone.zero_init_res = False  # True for SimSiam
cfg.model.backbone.dilation_list = [False, False, False]  # layer2 layer3 layer4, in increasing depth order
cfg.model.backbone.norm_layer = None  # None for frozenBN

cfg.model.backbone.top_layer = 'layer3'
cfg.model.backbone.use_inter_layer = False

cfg.model.backbone.lr_mult = 0.1
cfg.model.backbone.train_all = (cfg.model.backbone.lr_mult > 0) & False


# neck
cfg.model.neck = Edict()
cfg.model.neck.type = 'ModalFusion'
cfg.model.neck.in_channels_list = []
cfg.model.neck.inter_channels = 256
cfg.model.neck.num_tasks = 4
cfg.model.neck.use_language = cfg.model.use_language
cfg.model.neck.output_size = [16, 16]


cfg.model.neck.transformer = Edict()

cfg.model.neck.transformer.in_channels = 256
cfg.model.neck.transformer.num_heads = 8
cfg.model.neck.transformer.dim_feed = 2048
cfg.model.neck.transformer.dropout = 0.1
cfg.model.neck.transformer.activation = 'relu'
cfg.model.neck.transformer.norm_before = False

# default [6 0] for couple [4 4] for decouple
cfg.model.neck.transformer.num_encoders = 6
cfg.model.neck.transformer.num_decoders = 0
cfg.model.neck.transformer.return_inter_decode = False


# head
cfg.model.head = Edict()
cfg.model.head.type = ['ExtremeV3']  # multi-head list | 'LearnedToken' 'Extreme'
cfg.model.head.in_channels = cfg.model.neck.inter_channels
cfg.model.head.inter_channels = 256
cfg.model.head.output_size = [16, 16]


# criterion
cfg.model.criterion = Edict()
cfg.model.criterion.type = ['MultiModal']
cfg.model.criterion.alpha_giou = 2
cfg.model.criterion.alpha_l1 = 5
cfg.model.criterion.alpha_conf = 1


# ############################################
#                  data
# ############################################
cfg.data = Edict()

cfg.data.num_works = 8
cfg.data.batch_size = 32
cfg.data.sample_range = 200

cfg.data.datasets_train = []
cfg.data.datasets_val = []

cfg.data.num_samples_train = 60000
cfg.data.num_samples_val = 6000


cfg.data.search_size = 256
cfg.data.search_scale_f = 4.0
cfg.data.search_jitter_f = [0.5, 3]

cfg.data.template_size = 128
cfg.data.template_scale_f = 2.0
cfg.data.template_jitter_f = [0., 0.]  # [0.1, 0.5]


# ############################################
#                  trainer
# ############################################
cfg.trainer = Edict()

cfg.trainer.seed = 123
cfg.trainer.start_epoch = 0
cfg.trainer.end_epoch = 500
cfg.trainer.sync_bn = False
cfg.trainer.amp = False

cfg.trainer.resume = None
cfg.trainer.pretrain = None
cfg.trainer.pretrain_lr_mult = None

cfg.trainer.val_interval = 10
cfg.trainer.print_interval = 1
cfg.trainer.save_interval = 1


# distributed train
cfg.trainer.dist = Edict()
cfg.trainer.dist.distributed = False
cfg.trainer.dist.master_addr = None
cfg.trainer.dist.master_port = None

cfg.trainer.dist.device = 'cuda'
cfg.trainer.dist.world_size = None
cfg.trainer.dist.local_rank = None
cfg.trainer.dist.rank = None


# optimizer
cfg.trainer.optim = Edict()
cfg.trainer.optim.type = 'AdamW'
cfg.trainer.optim.weight_decay = 1e-4
cfg.trainer.optim.momentum = 0.9

cfg.trainer.optim.grad_clip_norm = 1.0

cfg.trainer.optim.expect_lr = 1e-4
cfg.trainer.optim.expect_batch_size = 128
cfg.trainer.optim.base_lr = cfg.trainer.optim.expect_lr * cfg.trainer.optim.expect_batch_size / 128
cfg.trainer.optim.grad_acc_steps = cfg.trainer.optim.expect_batch_size // 2 // cfg.data.batch_size


# lr_scheduler
cfg.trainer.lr_scheduler = Edict()
cfg.trainer.lr_scheduler.type = 'multi_step'  # lr_scheduler list | 'cosine' 'multi_step'
cfg.trainer.lr_scheduler.warmup_epoch = 0
cfg.trainer.lr_scheduler.milestones = [400]


if __name__ == '__main__':
    import pprint

    print('\n' + pprint.pformat(cfg))

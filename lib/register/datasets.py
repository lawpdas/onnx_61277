
data_register = dict()

# #################################################################################
data_register.update({
    'notnl2k': [
        ['vid_sent_train',  # 'refcocos_train',
         'got10k_train_vot', 'lasot_train', 'trackingnet_train_p0',
         'trackingnet_train_p1', 'trackingnet_train_p2'],
        ['got10k_val', 'tnl2k_test', 'vid_sent_val']
    ],
})

data_register.update({
    'language': [
        ['tnl2k_train', 'vid_sent_train',  # 'refcocos_train',
         'got10k_train_vot', 'lasot_train', 'trackingnet_train_p0',
         'trackingnet_train_p1', 'trackingnet_train_p2'],
        ['got10k_val', 'tnl2k_test', 'vid_sent_val']
    ],
})

data_register.update({
    'overall': [
        ['got10k_train_vot', 'lasot_train', 'trackingnet_train_p0',
         'trackingnet_train_p1', 'trackingnet_train_p2'],
        ['got10k_val']
    ],
})

data_register.update({
    'got10k': [
        ['got10k_train'],
        ['got10k_val']
    ],
})

data_register.update({
    'tnl2k': [
        ['tnl2k_train'],
        ['tnl2k_test']
    ],
})

data_register.update({
    'refer': [
        ['tnl2k_train', 'vid_sent_train'],
        ['got10k_val', 'tnl2k_test', 'vid_sent_val']
    ],
})

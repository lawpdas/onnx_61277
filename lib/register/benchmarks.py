from functools import partial
from lib.evaluator.benchmark_loaders import load_lasot, load_tnl2k, load_votlt, \
    load_otb_lang
from lib.register.paths import path_register as path

benchmark_register = dict()

# #################################################################################
benchmark_register.update({
    'got10k_test': NotImplemented
})

benchmark_register.update({
    'got10k_val': NotImplemented
})

benchmark_register.update({
    'lasot': partial(load_lasot, root=path.benchmark.lasot)
})

benchmark_register.update({
    'trackingnet': NotImplemented
})

benchmark_register.update({
    'tnl2k': partial(load_tnl2k, root=path.benchmark.tnl2k)
})

benchmark_register.update({
    'vot20': NotImplemented
})

benchmark_register.update({
    'vot2019lt': partial(load_votlt, root=path.benchmark.vot19lt)
})

benchmark_register.update({
    'vot2018lt': partial(load_votlt, root=path.benchmark.vot18lt)
})

benchmark_register.update({
    'otb99lang': partial(load_otb_lang, root=path.benchmark.otb99lang)
})

benchmark_register.update({
    'choices': list(benchmark_register.keys())
})

if __name__ == '__main__':
    print(benchmark_register['choices'])
    for k, load_fn in benchmark_register.items():
        if load_fn is NotImplemented:
            print(k, 'NotImplemented')
        else:
            load_fn()

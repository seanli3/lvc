from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('localWL')
def set_cfg_example(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    cfg.localWL = CN()

    cfg.localWL.hops = 3

    cfg.localWL.pool = 'sum'

    cfg.localWL.dropout = 0.8

    cfg.localWL.walk = 'bfs'

    cfg.localWL.sortBy = 'degree'

    cfg.localWL.reverse = False

    cfg.localWL.beamSize = None

    cfg.localWL.maxPathLen = 20

    cfg.localWL.hop_pool = 'cat'


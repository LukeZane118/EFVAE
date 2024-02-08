from .uniform_sampler import UniformSampler
from .popular_sampler import PopularSampler
# from .dns_sampler import DynamicNegativeSampler
from .fims_uni_sampler import FIMSUniSampler
from .fims_pop_sampler import FIMSPopSampler

def get_sampler(config, dataset):
    if config.sampler_name == 'Uni':
        sampler = UniformSampler(config, dataset)
    elif config.sampler_name == 'Pop':
        sampler = PopularSampler(config, dataset)
    # elif config.sampler_name == 'DNS':
    #     sampler = DynamicNegativeSampler(config, dataset)
    elif config.sampler_name == 'FIMS-Uni':
        # item_emb = self._get_item_emb().detach()
        sampler = FIMSUniSampler(config, dataset)
    elif config.sampler_name == 'FIMS-Pop':
        # item_emb = self._get_item_emb().detach()
        sampler = FIMSPopSampler(config, dataset)
    else:
        raise ValueError('Unknown sampler name.')
    return sampler
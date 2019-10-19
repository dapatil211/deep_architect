import importlib
import deep_architect.utils as ut


def split_module_string(module_string):
    module = module_string.split('.')
    return '.'.join(module[:-1]), module[-1]


def get_class(module_string):
    module, class_name = split_module_string(module_string)
    m = importlib.import_module(module)
    return getattr(m, class_name)


def create_search_space(search_space_config, exp_config):
    if isinstance(exp_config, str):
        exp_config = ut.read_jsonfile(exp_config)
    if isinstance(search_space_config, str):
        search_space_config = ut.read_jsonfile(search_space_config)

    ssf_class = get_class(search_space_config['location'])
    return ssf_class(exp_config=exp_config, **search_space_config['kw_args'])


def create_searcher(searcher_config, exp_config, ssf):
    if isinstance(exp_config, str):
        exp_config = ut.read_jsonfile(exp_config)
    if isinstance(searcher_config, str):
        searcher_config = ut.read_jsonfile(searcher_config)

    se_class = get_class(searcher_config['location'])
    return se_class(ssf, exp_config=exp_config, **searcher_config['kw_args'])


def create_evaluator(evaluator_config, exp_config, eval_kwargs):
    if isinstance(exp_config, str):
        exp_config = ut.read_jsonfile(exp_config)
    if isinstance(evaluator_config, str):
        evaluator_config = ut.read_jsonfile(evaluator_config)

    evaluator_class = get_class(evaluator_config['location'])
    base_args = evaluator_config['kw_args']
    base_args.update(eval_kwargs)
    return evaluator_class(exp_config=exp_config, **base_args)

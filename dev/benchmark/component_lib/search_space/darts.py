import ipdb
import deep_architect.core as co
import deep_architect.modules as mo
import itertools

st = ipdb.set_trace
"""
-> get_unassigned_hparam
-> assign all single values
-> select an hparam
-> go to a module
-> get h_params
-> create cross_product of unassigned + assigned
-> copy module with new hparam_dict
"""


def copy_module(m, hparams):
    module = type(m)
    name = '-'.join(m.get_name().split('-')[:-1])
    name = '.'.join(m.get_name().split('.')[1:])
    input_names = [name for name in m.inputs]
    output_names = [name for name in m.outputs]
    copy = module(name, m._compile_fn, hparams, input_names, output_names)
    return copy.inputs, copy.outputs


def create_hparam_cross_product(hparams):
    values = []
    for hparam in hparams:
        h = hparams[hparam]
        # Second part of condition shouldn't be triggered
        if h.has_value_assigned() or len(h.vs) == 1:
            values.append((hparam, [h.val]))
        else:
            values.append((hparam, h.vs))
    h_instantiations = itertools.product(*[[(value[0], value[1][i])
                                            for i in range(len(value[1]))]
                                           for value in values])
    return [dict(h_instantiation) for h_instantiation in h_instantiations]


# def disconnect_module(m):
#     for inp in m.inputs:
#         m.inputs[inp].


def dartsify(ss, combine_fn):
    ins, outs = ss

    while not co.is_specified(outs):
        print('here')
        hs = list(co.get_unassigned_independent_hyperparameters(outs))
        print(hs)
        for h in hs:
            if len(h.vs) == 1:
                h.assign_value(h.vs[0])
        hs = list(co.get_unassigned_independent_hyperparameters(outs))
        if len(hs) > 0:
            h = hs[0]

            module = list(h.modules)[0]
            module_hs = module.hyperps
            h_instantiations = create_hparam_cross_product(module_hs)
            # num_instantiations = len(h_instantiations)
            instantiated_modules = [
                copy_module(module, h_instantiation)
                for h_instantiation in h_instantiations
            ]

            c_ins, c_outs = combine_modules(instantiated_modules, combine_fn)
            # m_input_connections = {
            #     inp: module.inputs[inp].get_connected_output()
            #     for inp in module.inputs
            # }
            # m_output_connections = {
            #     out: module.outputs[out].get_connected_inputs()
            #     for out in module.outputs
            # }

            for inp in module.inputs:
                module.inputs[inp].reroute_connected_output(c_ins[inp])
            for out in module.outputs:
                module.outputs[out].reroute_all_connected_inputs(c_outs[out])

    return ins, outs


def combine_modules(modules, combine_fn):
    i_ins, i_outs = mo.identity()
    c_ins, c_outs = combine_fn(len(modules))
    for ix, (m_ins, m_outs) in enumerate(modules):
        i_outs['out'].connect(m_ins['in'])
        m_outs['out'].connect(c_ins['in%d' % ix])

    return i_ins, c_outs
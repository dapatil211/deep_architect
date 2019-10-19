"""
This is an implementation of the search space from the paper "Designing Neural
Network Architectures using Reinforcement Learning" (Baker et al). The search
space was written using descriptions from the paper
(https://arxiv.org/abs/1611.02167) and by looking at the implementation
released by the authors (https://github.com/bowenbaker/metaqnn).
"""
from __future__ import absolute_import
import math

from deep_architect.contrib.deep_learning_backend.tensorflow_keras_ops import (
    relu, batch_normalization, conv2d, max_pool2d, fc_layer, dropout, flatten,
    global_pool2d)

import deep_architect.modules as mo
import deep_architect.core as co
from deep_architect.hyperparameters import Discrete as D
from ..api import SearchSpaceFactory


def get_new_image_size(h_image_size, h_kernel, h_stride):
    return co.DependentHyperparameter(
        lambda dh: int(
            math.ceil(
                float(dh['image_size'] - dh['kernel'] + 1) / float(dh['stride'])
            )), {
                'image_size': h_image_size,
                'kernel': h_kernel,
                'stride': h_stride
            })


def get_layers(
        prev_layer,
        depth,
        h_image_size,
        num_classes,
        layers,
        prev_fc=0,
        fc_limit=2,
        depth_limit=6,
        add_batch_norm=False,
):
    possible_layers = ['softmax']
    if depth < depth_limit:
        if prev_layer == 'conv' or prev_layer == 'pool' or prev_layer == '':
            possible_layers.append('conv')
            possible_layers.append('fc')
            possible_layers.append('gap')
        if prev_layer == 'conv' or prev_layer == '':
            possible_layers.append('pool')
        if prev_layer == 'fc' and prev_fc < fc_limit:
            possible_layers.append('fc')
    h_image_size = [h_image_size]
    prev_fc = [prev_fc]

    def substitution_fn(dh):
        # h_image_size = h_image_size

        if dh['layer'] == 'conv':
            layer = conv2d(
                D([64, 128, 256, 512]),
                D([
                    kernel_size for kernel_size in [1, 3, 5]
                    if kernel_size < dh['image_size']
                ]))
            layer = [layer]
            if add_batch_norm:
                layer.append(batch_normalization())
            layer.append(relu())
            layer = mo.siso_sequential(layer)
        elif dh['layer'] == 'pool':
            h_kernel_size = D([
                kernel_size for kernel_size in [2, 3, 5]
                if kernel_size < dh['image_size']
            ])
            h_stride = co.DependentHyperparameter(
                lambda dh: 3 if dh['kernel_size'] == 5 else 2,
                {'kernel_size': h_kernel_size})
            layer = max_pool2d(h_kernel_size, h_stride, h_padding='VALID')
            h_image_size[0] = get_new_image_size(h_image_size[0], h_kernel_size,
                                                 h_stride)

            layer = [layer]
            if add_batch_norm:
                layer.append(batch_normalization())
            layer = mo.siso_sequential(layer)
        elif dh['layer'] == 'fc':
            layer = mo.siso_sequential([fc_layer(D([128, 256, 512])), relu()])
            prev_fc[0] += 1
        elif dh['layer'] == 'gap':
            layer = mo.siso_sequential(
                [conv2d(num_classes, 1),
                 relu(), global_pool2d()])
        else:
            layers.append(mo.siso_sequential([flatten(), fc_layer(10)]))
            return layers[-1]

        next_layers = get_layers(dh['layer'], depth + 1, h_image_size[0],
                                 num_classes, layers, prev_fc[0], fc_limit,
                                 depth_limit, add_batch_norm)
        layers.append(layer)
        total_layers = depth + len(layers)

        if depth % 2 == 0:
            layer = mo.siso_sequential(
                [layer, dropout(depth / (total_layers * 2.0))])
        return mo.siso_sequential([layer, next_layers])

    return mo.substitution_module('Layer', substitution_fn, {
        'layer': D(possible_layers),
        'image_size': h_image_size[0]
    }, ['in'], ['out'])


def get_search_space():

    x = get_layers(prev_layer='',
                   depth=0,
                   h_image_size=D([32]),
                   num_classes=10,
                   layers=[],
                   prev_fc=0,
                   fc_limit=2,
                   depth_limit=18,
                   add_batch_norm=False)
    return x


class Baker2016Designing_SSF(SearchSpaceFactory):

    def __init__(self, exp_config):
        SearchSpaceFactory.__init__(self, get_search_space, exp_config)

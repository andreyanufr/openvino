# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....samplers.creator import create_sampler
from ....statistics.statistics import TensorStatistic
from ....statistics.functions import activations as asf
from ....statistics.functions import aggregation as agf
from ....utils.logger import get_logger

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.front.common.partial_infer.utils import mo_array

logger = get_logger(__name__)


@COMPRESSION_ALGORITHMS.register('SmoothQuantize')
class SmoothQuantize(Algorithm):
    name = 'SmoothQuantize'

    def __init__(self, config, engine):
        super().__init__(config, engine)
        stat_subset_size = min(
            self._config.get(
                'stat_subset_size', len(self._engine.data_loader)),
            len(self._engine.data_loader))
        stat_batch_size = min(
            self._config.get('stat_batch_size', 1), len(self._engine.data_loader))
        self.total_exec_steps = stat_subset_size
        shuffle_data = self._config.get('shuffle_data', False)
        seed = self._config.get('seed', 0)
        self._sampler = create_sampler(
            engine, stat_subset_size, shuffle_data, seed, stat_batch_size)
        self.alpha = 0.5

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        """ this function applies smooth quantize procedure
         :param model: model to apply the algo on
         :return range-corrected model
         """

        activations_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        mat_mul_inputs = self.find_mat_muls(model)

        stats = dict()
        for node_name, stats_list in activations_statistics.items():
            stats[node_name] = dict()
            for stats_name, stats_values in stats_list.items():
                stats[node_name][stats_name] = agf.median(stats_values)

        for node_mat_mul_data in mat_mul_inputs:
            node_0, node_1, node_mat_mul = node_mat_mul_data
            self.smooth_quantize(model, node_0, node_1, node_mat_mul, stats)

        return model

    def smooth_quantize(self, model, node_0, node_1, node_mat_mul, stats):
        # node_0 * node_1 = node_mat_mul_out
        name_0 = node_0.fullname
        name_1 = node_1.fullname

        is_bmm = True

        if name_0 in stats:
            stats_0 = stats[name_0]['channel_range_max']
            stats_0 = np.max(stats_0, axis=0)
            # assert len(stats_0.shape) > 2
        else:
            # TODO: check if input_node[0] can be constant
            raise Exception("Input node 0 is Const")
            stats_0 = self.get_weights(node_0)
            is_bmm = False

        if name_1 in stats:
            stats_1 = stats[name_1]['channel_range_max']
        else:
            if node_1.type != 'Const':
                return
            stats_1 = deepcopy(nu.get_node_value(node_1))
            # TODO: check it
            # if node_mat_mul['transpose_b']:
            #     stats_1 = np.transpose(stats_1) # copy
            stats_1 = np.abs(stats_1)
            stats_1 = np.max(stats_1, axis=0)  # abs_max per column [M, K] * [K, N]
            is_bmm = False

        if is_bmm:
            self.smooth_quantize_activation_activation(node_0, node_1, node_mat_mul, stats_0, stats_1)
        else:
            self.smooth_quantize_activation_linear(model, node_0, node_1, node_mat_mul, stats_0, stats_1)

    def smooth_quantize_activation_activation(self, node_0, node_1, node_mat_mul, stats_0, stats_1):
        # node_0 * node_1 = node_mat_mul_out
        # TODO: theoretical place for improvement
        pass

    def smooth_quantize_activation_linear(self, model, node_a, node_l, node_mat_mul, stats_a, stats_l, ratio=0.5):
        # node_0 * node_1 = node_mat_mul_out, [M, K] * [K, N] = [M, N]

        # check it later
        # axis = ''
        # i_a_max = np.argmax(stats_a, axis=axis)
        # c_a_max = stats_a[i_a_max]
        #
        # i_a_min = np.argmax(stats_a, axis=axis)
        # c_a_min = stats_a[i_a_min]
        #
        # if c_a_min / c_a_max > ratio:
        #     # no reason to make smoothing
        #     return

        scales = (np.power(stats_a, self.alpha) / (np.power(stats_l, 1 - self.alpha) + np.finfo(float).eps))
        scales = np.clip(scales, a_min=1e-5, a_max=None)

        w_scales = np.expand_dims(scales, axis=0)
        weights = nu.get_node_value(node_l)  # self.get_weights(node_l)
        weights = weights * w_scales

        nu.set_node_value(node_l, weights)

        # scales = np.broadcast_to(scales, stats_a)
        scales = scales ** (-1)

        self.insert_multiply_node(node_a, node_mat_mul, scales)

        # TODO: need to insert Multiply node between node_a and node_mat_mul with scale**(-1)

        pass

    @staticmethod
    def insert_multiply_node(node_in, node_out, scales):
        # assume node_out.in_port(0) == node_in
        port_num = 0
        for i in range(len(node_in.out_ports())):
            destination_ports = []
            for dest_port in node_in.out_port(i).get_destinations():
                if node_out.in_port(0) is dest_port:
                    port_num = i
                destination_ports.append(dest_port)
            if node_out.in_port(0) in destination_ports:
                port_num = i
                break

        assert port_num == 0
        if port_num == -1:
            print("Something wrong in connection between {} and {}".format(node_in.name, node_out.name))
            return

        destination_ports = []
        for dest_port in node_in.out_port(0).get_destinations():
            destination_ports.append(dest_port)

        # ##node_out.graph.remove_edge(node_out.in_port(0).id, node_in.out_port(0).id)

        mul_node = Mul(node_out.graph, {'name': node_out.id + '/sq_mul', 'need_shape_inference': True}).create_node()
        const_node = Const(node_out.graph, {'name': node_out.id + '/sq', 'value': mo_array(scales)}).create_node()
        const_node.out_port(0).connect(mul_node.in_port(1))

        mul_node.out_port(0).connect(node_out.in_port(0))
        mul_node.out_port(0).get_connection().set_destination(node_out.in_port(0))

        node_in.out_port(0).disconnect()
        node_in.out_port(0).connect(mul_node.in_port(0))
        node_in.out_port(0).get_connection().set_destination(mul_node.in_port(0))

        for dest_port in destination_ports:
            if node_out.in_port(0) == dest_port:
                continue
            node_in.out_port(0).connect(dest_port)

    @staticmethod
    def get_weights(node):
        # get consumer convolution weights
        w_out = nu.get_weights_for_node(node)
        if w_out.type == 'FakeQuantize':
            w_out = nu.get_node_input(w_out, 0)
        if w_out.type != 'Const':
            w_out = None
            logger.debug('{} has no const weights. '
                         'Do not align activations for this node pair.'.format(node.fullname))

        return w_out

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        activation_statistics_layout = self.get_activations_statistics_layout(model)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler)
        self._stats_collector = stats_collector

    def get_activations_statistics_layout(self, model):
        node_pairs_list = self.find_mat_muls(model)

        stats_layout = {}
        for node_pair_data in node_pairs_list:
            node_in, *_ = node_pair_data
            # Step over bias Add node
            if nu.get_bias_for_node(node_in):
                node_in = nu.get_node_output(node_in, 0)[0]
            name = node_in.fullname
            stats_layout[name] = {'channel_range_min': TensorStatistic(asf.abs_max_per_channel_transformer),
                                  'channel_range_max': TensorStatistic(asf.abs_max_per_channel_transformer)}

        logger.debug('Collecting output statistics for nodes {}'.format(stats_layout.keys()))
        return stats_layout

    def find_mat_muls(self, model):
        node_pairs_list = []
        nodes = sorted([(n.fullname, n) for n in mu.get_nodes_by_type(model, ['MatMul'])])
        for _, node_out in nodes:
            node_0 = nu.get_node_input(node_out, 0)
            node_1 = nu.get_node_input(node_out, 1)  # Const or other activation

            node_pairs_list.append((node_0, node_1, node_out))
            # try smooth quantize inside this sequence
            logger.debug('{} -> {}'.format(node_0.fullname, node_1.fullname))

        return node_pairs_list

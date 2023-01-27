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

        for node_pair_data in mat_mul_inputs:
            pass

        return model

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        activation_statistics_layout = self.get_activations_statistics_layout(model)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler)
        self._stats_collector = stats_collector

    def get_activations_statistics_layout(self, model):
        node_pairs_list = self.find_node_pairs(model)

        stats_layout = {}
        for node_pair_data in node_pairs_list:
            node_in, *_ = node_pair_data
            # Step over bias Add node
            if nu.get_bias_for_node(node_in):
                node_in = nu.get_node_output(node_in, 0)[0]
            name = node_in.fullname
            stats_layout[name] = {'channel_range_min': TensorStatistic(asf.quantile_per_channel, q=1e-4),
                                  'channel_range_max': TensorStatistic(asf.quantile_per_channel, q=1-1e-4)}

        logger.debug('Collecting output statistics for nodes {}'.format(stats_layout.keys()))
        return stats_layout

    def find_mat_muls(self, model):
        node_pairs_list = []
        nodes = sorted([(n.fullname, n) for n in mu.get_nodes_by_type(model, ['MatMul'])])
        for _, node_out in nodes:
            node_0 = nu.get_node_input(node_out, 0)
            node_1 = nu.get_node_input(node_out, 1) # Const or other activation

            node_pairs_list.append((node_0, node_1))
            # try smooth quantize inside this sequence
            logger.debug('{} -> {}'.format(node_0.fullname, node_1.fullname))

        return node_pairs_list

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import os
import sys
import numpy as np
from numpy import linalg as LA

np.set_printoptions(threshold=sys.maxsize)

from ...algorithm import Algorithm
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ....graph import model_utils as mu
from ....graph import node_utils as nu
from ....samplers.creator import create_sampler
from ....statistics.statistics import TensorStatistic
from ....statistics.functions import activations as asf
from ....statistics.functions import aggregation as agf
from ....utils.logger import get_logger
from ..fake_quantize import fix_zero_filters_symmetric

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.runtime import opset9
import openvino.runtime as ov
from openvino.runtime import Core

logger = get_logger(__name__)


# borrow from fake_quantize
def symmetric_range(max_level, min_level, is_weight=True):
    levels = 256
    if is_weight:
        max_level = fix_zero_filters_symmetric(max_level)
        min_level = -max_level
    else:
        max_level = fix_zero_filters_symmetric(max_level)
        signed = True
        min_level = np.zeros(max_level.shape) if np.all(min_level >= 0) and not signed else \
            -max_level * levels / (levels - 2)
    # min_level, max_level = broadcast_fq_values(fq, node, min_level, max_level, fake_quantize_config)
    return max_level, min_level


def create_quantizer(data_shape, input_low, input_high, output_low, output_high):
    core = Core()
    data = opset9.parameter(data_shape)
    input_low_f = opset9.constant(input_low)
    input_high_f = opset9.constant(input_high)
    output_low_f = opset9.constant(output_low)
    output_high_f = opset9.constant(output_high)
    levels = 256

    op = opset9.fake_quantize(data, input_low_f, input_high_f,
                              output_low_f, output_high_f, levels)

    r = opset9.result(op)

    model = ov.Model([r], [data])

    executable_network = core.compile_model(model, device_name='CPU')

    return executable_network


def create_matmul_layer_int8(data_shape,
                             input_low_w, input_high_w,
                             input_low_a, input_high_a,
                             weight,
                             multiply=None):
    core = Core()
    data = opset9.parameter(data_shape)

    input_low_a_op = opset9.constant(input_low_a)
    input_high_a_op = opset9.constant(input_high_a)
    output_low_a_op = opset9.constant(input_low_a)
    output_high_a_op = opset9.constant(input_high_a)
    levels = 256

    if multiply is not None:
        scale = opset9.constant(multiply)
        mul = opset9.multiply(data, scale)
        fq_a = opset9.fake_quantize(mul, input_low_a_op, input_high_a_op,
                                    output_low_a_op, output_high_a_op, levels)
    else:
        fq_a = opset9.fake_quantize(data, input_low_a_op, input_high_a_op,
                                    output_low_a_op, output_high_a_op, levels)

    input_w = opset9.constant(weight)

    input_low_w = np.expand_dims(input_low_w, axis=1)
    input_high_w = np.expand_dims(input_high_w, axis=1)
    input_low_w_op = opset9.constant(input_low_w)
    input_high_w_op = opset9.constant(input_high_w)
    output_low_w_op = opset9.constant(input_low_w)
    output_high_w_op = opset9.constant(input_high_w)

    fq_w = opset9.fake_quantize(input_w, input_low_w_op, input_high_w_op,
                                output_low_w_op, output_high_w_op, levels)

    mat_mul = opset9.matmul(fq_a, fq_w, transpose_a=False, transpose_b=True)

    r = opset9.result(mat_mul)

    model = ov.Model([r], [data])

    executable_network = core.compile_model(model, device_name='CPU')

    return executable_network, model


def create_matmul_layer_fp32(data_shape, weight):
    core = Core()
    data = opset9.parameter(data_shape)

    input_w = opset9.constant(weight)

    mat_mul = opset9.matmul(data, input_w, transpose_a=False, transpose_b=True)

    r = opset9.result(mat_mul)

    model = ov.Model([r], [data])

    executable_network = core.compile_model(model, device_name='CPU')

    return executable_network


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
        self.alpha = self._config.get('alpha', 0.95)
        self.history = {}

        self.layer_number = self._config.get('layer_number', -1)
        self.use_grid_alpha = self._config.get('use_grid_alpha', False)

        print("self.layer_number ", self.layer_number)
        print("self.alpha ", self.alpha)

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        """ this function applies smooth quantize procedure
         :param model: model to apply the algo on
         :return range-corrected model
         """

        activations_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        mat_mul_inputs = self.find_mat_muls_filtered(model)

        stats = dict()
        for node_name, stats_list in activations_statistics.items():
            stats[node_name] = dict()
            stats[node_name]['channel_range_max'] = agf.max_per_channel(stats_list['channel_range_max'])
            stats[node_name]['channel_range_mean'] = np.mean(stats_list['channel_range_mean'], axis=0)
            if self.use_grid_alpha:
                stats[node_name]['channel_range_mean_max'] = stats_list['channel_range_max']

        print("Process {} MatMul nodes".format(len(mat_mul_inputs)))

        mat_mul_inputs_groupped = self.group_nodes_by_source(mat_mul_inputs)

        for k, val in mat_mul_inputs_groupped.items():
            if self.use_grid_alpha:
                self.smooth_quantize_groupped_grid_alpha(k, val, stats)
            else:
                self.smooth_quantize_groupped_fixed_alpha(k, val, stats)

        return model

    @staticmethod
    def group_nodes_by_source(mat_mul_inputs):
        res = {}
        for node_mat_mul_data in mat_mul_inputs:
            node_0, node_1, node_mat_mul = node_mat_mul_data

            if node_0 in res:
                res[node_0].append((node_1, node_mat_mul))
            else:
                res[node_0] = [(node_1, node_mat_mul)]

        return res

    def smooth_quantize_groupped_best_conditioned_alpha(self, node_in, dst_nodes, stats):
        # node_0 * node_1 = node_mat_mul_out
        name_in = node_in.fullname
        bmms = 0

        if not name_in in stats:
            raise Exception("Input node 0 is Const")

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node
            name_in_1 = node_in_1.fullname
            if name_in_1 in stats or node_in_1.type != 'Const':
                bmms += 1

        if bmms > 0:  # MatMul with other activation
            return

        stats_activation = stats[name_in]['channel_range_max']
        stats_activation = np.clip(stats_activation, a_min=1e-5, a_max=None)

        if stats_activation.size <= 1:
            print(f"Skip node {name_in} smooth quantize. Shape: ", stats_activation.shape)
            return

        print(f"Node {name_in} smooth quantize. Shape: ", stats_activation.shape)

        alpha_grid = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        alpha_scales = {}

        for alpha in alpha_grid:
            best_scale = None
            best_ratio = 0.0
            # compute the most smooth scale in the case of |MatMuls| > 1
            for dst_node in dst_nodes:
                node_in_1, node_mat_mul = dst_node

                if node_in_1.type != 'Const':
                    raise Exception("Wrong type of FC node")

                stats_w = deepcopy(nu.get_node_value(node_in_1))

                name_in_1 = node_in_1.fullname

                if not node_mat_mul['transpose_b']:
                    raise Exception("Bad transpose")

                stats_w = np.abs(stats_w)
                stats_w = np.max(stats_w, axis=0)  # abs_max per column [M, K] * [K, N]
                stats_w = np.clip(stats_w, a_min=1e-5, a_max=None)

                scales = (np.power(stats_activation, alpha) / (np.power(stats_w, 1 - alpha) + np.finfo(float).eps))
                a_min = np.quantile(scales, 0.1)
                scales = np.clip(scales, a_min=a_min, a_max=1e2)

                ratio = scales.min() / scales.max()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_scale = deepcopy(scales)

            a_scales = best_scale ** (-1)
            w_scales = np.expand_dims(best_scale, axis=0)
            alpha_scales[alpha] = (a_scales, w_scales)

        best_alpha = -1.0
        smallest_error = 1000.0

        for alpha, (_, w_scale) in alpha_scales.items():
            int8_diff = 0.0
            int8_sq_diff = 0.0

            for dst_node in dst_nodes:
                node_in_1, node_mat_mul = dst_node

                weights = nu.get_node_value(node_in_1)
                scaled_weights = weights * w_scale

                diff_int8 = LA.cond(weights)
                diff_int8_sq = LA.cond(scaled_weights)

                int8_diff += diff_int8
                int8_sq_diff += diff_int8_sq

            if int8_sq_diff < smallest_error and int8_sq_diff < int8_diff:
                smallest_error = int8_sq_diff
                best_alpha = alpha

        if best_alpha < 0.0:
            print(f"Skip inserting of SQ for layer {name_in}")
            return
        print("Best alpha: ", best_alpha)

        a_scales, w_scales = alpha_scales[best_alpha]

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node

            weights = nu.get_node_value(node_in_1)
            weights = weights * w_scales
            nu.set_node_value(node_in_1, weights)

        a_scales = np.expand_dims(a_scales, axis=(0, 1))

        self.insert_multiply_node(node_in, dst_nodes, a_scales)

    def compute_mean_abs_max_stat(self, max_per_channel_vals, scale=1):
        vals = np.array([np.max(arr * scale) for arr in max_per_channel_vals])
        return np.mean(vals)

    def smooth_quantize_groupped_grid_alpha(self, node_in, dst_nodes, stats):
        # node_0 * node_1 = node_mat_mul_out
        name_in = node_in.fullname
        bmms = 0

        def infer(data, model):
            inference_request = model.create_infer_request()
            inference_request.infer(inputs=[data])
            result = inference_request.get_output_tensor(0).data
            return result

        if not name_in in stats:
            raise Exception("Input node 0 is Const")

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node
            name_in_1 = node_in_1.fullname
            if name_in_1 in stats or node_in_1.type != 'Const':
                bmms += 1

        if bmms > 0:  # MatMul with other activation
            return

        stats_activation = stats[name_in]['channel_range_max']
        stats_activation = np.clip(stats_activation, a_min=1e-5, a_max=None)

        if stats_activation.size <= 1:
            print(f"Skip node {name_in} smooth quantize. Shape: ", stats_activation.shape)
            return

        print(f"Node {name_in} smooth quantize. Shape: ", stats_activation.shape)

        alpha_grid = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        # alpha_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        # alpha_grid = [0.8, 0.9, 0.95]
        alpha_scales = {}

        transpose_a = False
        for alpha in alpha_grid:
            best_scale = None
            best_ratio = 0.0
            # compute the most smooth scale in the case of |MatMuls| > 1
            for dst_node in dst_nodes:
                node_in_1, node_mat_mul = dst_node

                if node_in_1.type != 'Const':
                    raise Exception("Wrong type of FC node")

                transpose_a = node_mat_mul['transpose_a']

                stats_w = deepcopy(nu.get_node_value(node_in_1))

                name_in_1 = node_in_1.fullname

                if not node_mat_mul['transpose_b']:
                    raise Exception("Bad transpose")

                stats_w = np.abs(stats_w)
                stats_w = np.max(stats_w, axis=0)  # abs_max per column [M, K] * [K, N]
                stats_w = np.clip(stats_w, a_min=1e-5, a_max=None)

                if np.size(stats_activation) != np.size(stats_w):
                    print('Activations and weights shape mismaths for layer ', name_in, stats_activation.shape,
                          stats_w.shape)
                    return

                scales = (np.power(stats_activation, alpha) / (np.power(stats_w, 1 - alpha) + np.finfo(float).eps))
                a_min = np.quantile(scales, 0.1)
                scales = np.clip(scales, a_min=a_min, a_max=1e2)

                ratio = scales.min() / scales.max()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_scale = deepcopy(scales)

            scales = best_scale ** (-1)
            w_scales = np.expand_dims(best_scale, axis=0)
            alpha_scales[alpha] = (scales, w_scales)

        mean_activation = stats[name_in]['channel_range_mean']  #
        max_activation = stats[name_in]['channel_range_max']
        mean_max_activation = stats[name_in]['channel_range_mean_max']

        while len(mean_activation.shape) < 3:
            mean_activation = np.expand_dims(mean_activation, axis=0)
        while len(max_activation.shape) < 3:
            max_activation = np.expand_dims(max_activation, axis=0)

        best_alpha = -1.0
        smallest_error = 1000.0

        for alpha, (a_scale, w_scale) in alpha_scales.items():
            int8_diff = 0.0
            int8_sq_diff = 0.0

            for dst_node in dst_nodes:
                node_in_1, node_mat_mul = dst_node

                weights = nu.get_node_value(node_in_1)
                fp32_model = create_matmul_layer_fp32(mean_activation.shape, weights)

                stats_w = np.abs(weights)
                input_hi_w = np.max(stats_w, axis=1)
                input_low_w = -input_hi_w
                input_hi_w, input_low_w = symmetric_range(input_hi_w, input_low_w, True)

                input_hi_a = self.compute_mean_abs_max_stat(
                    mean_max_activation)  # np.max(mean_max_activation)#np.max(mean_max_activation)
                input_low_a = -input_hi_a
                input_hi_a, input_low_a = symmetric_range(input_hi_a, input_low_a, False)

                int8_model, _ = create_matmul_layer_int8(mean_activation.shape,
                                                         input_low_w, input_hi_w,
                                                         input_low_a, input_hi_a,
                                                         weights)
                # tmp_name = name_in.replace('/', '_')
                # ov.serialize(ov_model, f'subgraphs/{tmp_name}.xml', f'subgraphs/{tmp_name}.bin')

                input_hi_a = self.compute_mean_abs_max_stat(mean_max_activation,
                                                            a_scale)  ##np.max(mean_max_activation * a_scale)#np.max(mean_max_activation * a_scale)
                input_low_a = -input_hi_a
                input_hi_a, input_low_a = symmetric_range(input_hi_a, input_low_a, False)

                scaled_weights = weights * w_scale
                stats_w = np.abs(scaled_weights)
                input_hi_w = np.max(stats_w, axis=1)
                input_low_w = -input_hi_w
                input_hi_w, input_low_w = symmetric_range(input_hi_w, input_low_w, True)

                int8_sq_model, _ = create_matmul_layer_int8(mean_activation.shape,
                                                            input_low_w, input_hi_w,
                                                            input_low_a, input_hi_a,
                                                            scaled_weights, multiply=a_scale)

                # tmp_name = name_in.replace('/', '_')
                # ov.serialize(ov_model_sq, f'subgraphs/{tmp_name}_sq.xml', f'subgraphs/{tmp_name}_sq.bin')

                fp32_out = infer(mean_activation, fp32_model)
                int8_out = infer(mean_activation, int8_model)
                int8_sq_out = infer(mean_activation, int8_sq_model)

                diff_int8 = np.mean(np.abs(fp32_out - int8_out))
                diff_int8_sq = np.mean(np.abs(fp32_out - int8_sq_out))

                int8_diff += diff_int8
                int8_sq_diff += diff_int8_sq

                # fp32_out    = infer(max_activation, fp32_model)
                # int8_out    = infer(max_activation, int8_model)
                # int8_sq_out = infer(max_activation, int8_sq_model)

                # diff_int8 = np.mean(np.abs(fp32_out - int8_out))
                # diff_int8_sq = np.mean(np.abs(fp32_out - int8_sq_out))

                int8_diff += diff_int8
                int8_sq_diff += diff_int8_sq

            if int8_sq_diff <= smallest_error and int8_sq_diff < int8_diff:
                smallest_error = int8_sq_diff
                best_alpha = alpha

        if best_alpha < 0.0:
            print(f"Skip inserting of SQ for layer {name_in}")
            return
        print("Best alpha: ", best_alpha)

        a_scales, w_scales = alpha_scales[best_alpha]

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node

            weights = nu.get_node_value(node_in_1)
            weights = weights * w_scales
            nu.set_node_value(node_in_1, weights)

        # scales = np.expand_dims(scales, axis=(0, 1))

        if transpose_a:
            a_scales = np.expand_dims(a_scales, axis=(0, 2))
        else:
            a_scales = np.expand_dims(a_scales, axis=(0, 1))

        self.insert_multiply_node(node_in, dst_nodes, a_scales)

    def smooth_quantize_groupped_fixed_alpha(self, node_in, dst_nodes, stats):
        # node_0 * node_1 = node_mat_mul_out
        name_in = node_in.fullname
        bmms = 0

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node
            name_in_1 = node_in_1.fullname
            if name_in_1 in stats or node_in_1.type != 'Const':
                bmms += 1

        if bmms > 0:  # we can not applay SQ for batch matrix multiplication
            return

        stats_activation = stats[name_in]['channel_range_max']
        stats_activation = np.clip(stats_activation, a_min=1e-5, a_max=None)

        if stats_activation.size <= 1:  # something wrong with actiavtions stats
            return

        logger.debug(f"Node {name_in} smooth quantize. Shape: {stats_activation.shape}")

        # compute the most smooth scale in the case of |MatMuls| > 1
        transpose_a = False
        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node

            if node_in_1.type != 'Const':
                raise Exception("Wrong type of FC node")
            transpose_a = node_mat_mul['transpose_a']

            stats_w = deepcopy(nu.get_node_value(node_in_1))

            name_in_1 = node_in_1.fullname

            if not node_mat_mul['transpose_b']:
                raise Exception("Wrong value transpose_b. Not implemented.")

            stats_w = np.abs(stats_w)
            stats_w = np.max(stats_w, axis=0)  # abs_max per column [M, K] * [K, N]
            stats_w = np.clip(stats_w, a_min=1e-5, a_max=None)

            if np.size(stats_activation) != np.size(stats_w):
                logger.debug(
                    f'Activations and weights shape mismaths for layer {name_in} {stats_activation.shape} and {stats_w.shape}')
                return
            scales = (np.power(stats_activation, self.alpha) / (
                        np.power(stats_w, 1 - self.alpha) + np.finfo(float).eps))
            a_min = np.quantile(scales, 0.1)
            scales = np.clip(scales, a_min=a_min, a_max=1e2)

            ratio = scales.min() / scales.max()

            if ratio > best_ratio:
                best_ratio = ratio
                best_scale = deepcopy(scales)

        a_scales = best_scale ** (-1)
        w_scales = np.expand_dims(best_scale, axis=0)

        for dst_node in dst_nodes:
            node_in_1, node_mat_mul = dst_node

            weights = nu.get_node_value(node_in_1)
            weights = weights * w_scales
            nu.set_node_value(node_in_1, weights)

        if transpose_a:
            a_scales = np.expand_dims(a_scales, axis=(0, 2))
        else:
            a_scales = np.expand_dims(a_scales, axis=(0, 1))

        self.insert_multiply_node(node_in, dst_nodes, a_scales)

    def smooth_quantize_activation_activation(self, node_0, node_1, node_mat_mul, stats_0, stats_1):
        # node_0 * node_1 = node_mat_mul_out
        # TODO: theoretical place for improvement
        print("MatMul {} is BMM.".format(node_mat_mul.name))
        return 0

    @staticmethod
    def insert_multiply_node(node_in, dst_nodes, scales):
        destination_ports = []
        for dest_port in node_in.out_port(0).get_destinations():
            destination_ports.append(dest_port)

        node_name = node_in.id + '/sq_mul'
        mul_node = Mul(node_in.graph, {'name': node_name, 'need_shape_inference': True}).create_node()
        mul_node['fullname'] = node_name
        mul_node['any_name'] = node_name
        mul_node['id'] = node_name
        mul_node['name'] = node_name

        const_node = Const(node_in.graph, {'name': node_in.id + '/sq', 'value': mo_array(scales)}).create_node()
        const_node['fullname'] = node_in.id + '/sq'
        const_node.out_port(0).connect(mul_node.in_port(1))

        node_in.out_port(0).disconnect()
        node_in.out_port(0).connect(mul_node.in_port(0))

        connected_ports = []
        for dst_node in dst_nodes:
            _, node_mat_mul = dst_node
            mul_node.out_port(0).connect(node_mat_mul.in_port(0))

            connected_ports.append(node_mat_mul.in_port(0))

        for dest_port in destination_ports:
            if dest_port in connected_ports:
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
            node_in, _, node_out = node_pair_data
            # Step over bias Add node
            if nu.get_bias_for_node(node_in):
                node_in = nu.get_node_output(node_in, 0)[0]
            name = node_in.fullname
            axis = 1 if node_out['transpose_a'] == True else 0

            stats_layout[name] = {'channel_range_mean': TensorStatistic(asf.mean_per_channel_transformer, axis=axis),
                                  'channel_range_max': TensorStatistic(asf.abs_max_per_channel_transformer, axis=axis)}

        logger.debug('Collecting output statistics for nodes {}'.format(stats_layout.keys()))
        return stats_layout

    def find_mat_muls_filtered(self, model):
        node_pairs_list = []

        candidates = {}
        skip_values = set()

        nodes = sorted([(n.fullname, n) for n in mu.get_nodes_by_type(model, ['MatMul'])])
        for _, node_out in nodes:
            node_0 = nu.get_node_input(node_out, 0)
            node_1 = nu.get_node_input(node_out, 1)  # Const or other activation

            if node_0.fullname in skip_values:
                continue

            if node_1.type != 'Const':  # skip matmul with no const
                skip_values.add(node_0.fullname)
                continue
            else:
                outs = nu.get_node_output_ports(node_1)
                if len(outs) > 1:  # skip const with more than one consumer
                    skip_values.add(node_0.fullname)
                    continue
            if node_0.fullname in candidates:
                candidates[node_0.fullname].append((node_0, node_1, node_out))
            else:
                candidates[node_0.fullname] = [(node_0, node_1, node_out)]
            # try smooth quantize inside this sequence
            logger.debug('{} -> {}'.format(node_0.fullname, node_1.fullname))

        for key, val in candidates.items():
            if key in skip_values:
                continue
            node_pairs_list.extend(val)

        return node_pairs_list

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

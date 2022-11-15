# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.fakequantize import FakeQuantize
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class FakeQuantizeFrontExtractor(FrontExtractorOp):
    op = 'FakeQuantize'
    enabled = True

    @classmethod
    def extract(cls, node):
        levels = onnx_attr(node, 'levels', 'i')
        FakeQuantize.update_node_stat(node, {'levels': levels})
        return FakeQuantizeFrontExtractor.enabled

class ConvertFP8FrontExtractor(FrontExtractorOp):
    op = 'ConvertFP8'
    enabled = True

    @classmethod
    def extract(cls, node):
        return ConvertFP8FrontExtractor.enabled

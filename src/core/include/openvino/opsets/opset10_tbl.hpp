// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

_OPENVINO_OP_REG(Abs, ov::op::v0)
_OPENVINO_OP_REG(Acos, ov::op::v0)
_OPENVINO_OP_REG(Add, ov::op::v1)
_OPENVINO_OP_REG(Asin, ov::op::v0)
_OPENVINO_OP_REG(Atan, ov::op::v0)
_OPENVINO_OP_REG(AvgPool, ov::op::v1)
_OPENVINO_OP_REG(BatchNormInference, ov::op::v5)
_OPENVINO_OP_REG(BinaryConvolution, ov::op::v1)
_OPENVINO_OP_REG(Broadcast, ov::op::v3)
_OPENVINO_OP_REG(Bucketize, ov::op::v3)
_OPENVINO_OP_REG(CTCGreedyDecoder, ov::op::v0)
_OPENVINO_OP_REG(Ceiling, ov::op::v0)
_OPENVINO_OP_REG(Clamp, ov::op::v0)
_OPENVINO_OP_REG(Concat, ov::op::v0)
_OPENVINO_OP_REG(Constant, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ConvertLike, ov::op::v1)
_OPENVINO_OP_REG(Convolution, ov::op::v1)
_OPENVINO_OP_REG(ConvolutionBackpropData, ov::op::v1)
_OPENVINO_OP_REG(Cos, ov::op::v0)
_OPENVINO_OP_REG(Cosh, ov::op::v0)
_OPENVINO_OP_REG(CumSum, ov::op::v0)
_OPENVINO_OP_REG(DeformablePSROIPooling, ov::op::v1)
_OPENVINO_OP_REG(DepthToSpace, ov::op::v0)
_OPENVINO_OP_REG(Divide, ov::op::v1)
_OPENVINO_OP_REG(Elu, ov::op::v0)
_OPENVINO_OP_REG(Erf, ov::op::v0)
_OPENVINO_OP_REG(Equal, ov::op::v1)
_OPENVINO_OP_REG(Exp, ov::op::v0)
_OPENVINO_OP_REG(ExtractImagePatches, ov::op::v3)
_OPENVINO_OP_REG(FakeQuantize, ov::op::v0)
_OPENVINO_OP_REG(Floor, ov::op::v0)
_OPENVINO_OP_REG(FloorMod, ov::op::v1)
_OPENVINO_OP_REG(GatherTree, ov::op::v1)
_OPENVINO_OP_REG(Greater, ov::op::v1)
_OPENVINO_OP_REG(GreaterEqual, ov::op::v1)
_OPENVINO_OP_REG(GridSample, ov::op::v9)
_OPENVINO_OP_REG(GroupConvolution, ov::op::v1)
_OPENVINO_OP_REG(GroupConvolutionBackpropData, ov::op::v1)
_OPENVINO_OP_REG(GRN, ov::op::v0)
_OPENVINO_OP_REG(HardSigmoid, ov::op::v0)
_OPENVINO_OP_REG(Less, ov::op::v1)
_OPENVINO_OP_REG(LessEqual, ov::op::v1)
_OPENVINO_OP_REG(Log, ov::op::v0)
_OPENVINO_OP_REG(LogicalAnd, ov::op::v1)
_OPENVINO_OP_REG(LogicalNot, ov::op::v1)
_OPENVINO_OP_REG(LogicalOr, ov::op::v1)
_OPENVINO_OP_REG(LogicalXor, ov::op::v1)
_OPENVINO_OP_REG(LRN, ov::op::v0)
_OPENVINO_OP_REG(LSTMCell, ov::op::v4)
_OPENVINO_OP_REG(MatMul, ov::op::v0)
_OPENVINO_OP_REG(Maximum, ov::op::v1)
_OPENVINO_OP_REG(Minimum, ov::op::v1)
_OPENVINO_OP_REG(Mod, ov::op::v1)
_OPENVINO_OP_REG(Multiply, ov::op::v1)
_OPENVINO_OP_REG(Negative, ov::op::v0)
_OPENVINO_OP_REG(NormalizeL2, ov::op::v0)
_OPENVINO_OP_REG(NotEqual, ov::op::v1)
_OPENVINO_OP_REG(OneHot, ov::op::v1)
_OPENVINO_OP_REG(PRelu, ov::op::v0)
_OPENVINO_OP_REG(PSROIPooling, ov::op::v0)
_OPENVINO_OP_REG(Pad, ov::op::v1)
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Power, ov::op::v1)
_OPENVINO_OP_REG(PriorBoxClustered, ov::op::v0)
_OPENVINO_OP_REG(Proposal, ov::op::v4)
_OPENVINO_OP_REG(Range, ov::op::v4)
_OPENVINO_OP_REG(Relu, ov::op::v0)
_OPENVINO_OP_REG(ReduceMax, ov::op::v1)
_OPENVINO_OP_REG(ReduceLogicalAnd, ov::op::v1)
_OPENVINO_OP_REG(ReduceLogicalOr, ov::op::v1)
_OPENVINO_OP_REG(ReduceMean, ov::op::v1)
_OPENVINO_OP_REG(ReduceMin, ov::op::v1)
_OPENVINO_OP_REG(ReduceProd, ov::op::v1)
_OPENVINO_OP_REG(ReduceSum, ov::op::v1)
_OPENVINO_OP_REG(RegionYolo, ov::op::v0)
_OPENVINO_OP_REG(ReorgYolo, ov::op::v0)
_OPENVINO_OP_REG(Reshape, ov::op::v1)
_OPENVINO_OP_REG(Result, ov::op::v0)
_OPENVINO_OP_REG(ReverseSequence, ov::op::v0)
_OPENVINO_OP_REG(ROIPooling, ov::op::v0)
_OPENVINO_OP_REG(ScatterNDUpdate, ov::op::v3)
_OPENVINO_OP_REG(Select, ov::op::v1)
_OPENVINO_OP_REG(Selu, ov::op::v0)
_OPENVINO_OP_REG(Sign, ov::op::v0)
_OPENVINO_OP_REG(Sigmoid, ov::op::v0)
_OPENVINO_OP_REG(Sin, ov::op::v0)
_OPENVINO_OP_REG(Sinh, ov::op::v0)
_OPENVINO_OP_REG(Sqrt, ov::op::v0)
_OPENVINO_OP_REG(SpaceToDepth, ov::op::v0)
_OPENVINO_OP_REG(Split, ov::op::v1)
_OPENVINO_OP_REG(SquaredDifference, ov::op::v0)
_OPENVINO_OP_REG(Squeeze, ov::op::v0)
_OPENVINO_OP_REG(StridedSlice, ov::op::v1)
_OPENVINO_OP_REG(Subtract, ov::op::v1)
_OPENVINO_OP_REG(Tan, ov::op::v0)
_OPENVINO_OP_REG(Tanh, ov::op::v0)
_OPENVINO_OP_REG(TensorIterator, ov::op::v0)
_OPENVINO_OP_REG(Tile, ov::op::v0)
_OPENVINO_OP_REG(Transpose, ov::op::v1)
_OPENVINO_OP_REG(Unsqueeze, ov::op::v0)
_OPENVINO_OP_REG(VariadicSplit, ov::op::v1)
_OPENVINO_OP_REG(ConvertFP8, ov::op::v1)

// New operations added in opset2
_OPENVINO_OP_REG(BatchToSpace, ov::op::v1)
_OPENVINO_OP_REG(SpaceToBatch, ov::op::v1)

// New operations added in opset3
_OPENVINO_OP_REG(EmbeddingBagPackedSum, ov::op::v3)
_OPENVINO_OP_REG(EmbeddingSegmentsSum, ov::op::v3)
_OPENVINO_OP_REG(EmbeddingBagOffsetsSum, ov::op::v3)
_OPENVINO_OP_REG(GRUCell, ov::op::v3)
_OPENVINO_OP_REG(NonZero, ov::op::v3)
_OPENVINO_OP_REG(RNNCell, ov::op::v0)
_OPENVINO_OP_REG(ScatterElementsUpdate, ov::op::v3)
_OPENVINO_OP_REG(ScatterUpdate, ov::op::v3)
_OPENVINO_OP_REG(ShuffleChannels, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v3)
_OPENVINO_OP_REG(TopK, ov::op::v3)

// New operations added in opset4
_OPENVINO_OP_REG(Acosh, ov::op::v3)
_OPENVINO_OP_REG(Asinh, ov::op::v3)
_OPENVINO_OP_REG(Atanh, ov::op::v3)
_OPENVINO_OP_REG(CTCLoss, ov::op::v4)
_OPENVINO_OP_REG(HSwish, ov::op::v4)
_OPENVINO_OP_REG(Interpolate, ov::op::v4)
_OPENVINO_OP_REG(Mish, ov::op::v4)
_OPENVINO_OP_REG(ReduceL1, ov::op::v4)
_OPENVINO_OP_REG(ReduceL2, ov::op::v4)
_OPENVINO_OP_REG(SoftPlus, ov::op::v4)
_OPENVINO_OP_REG(Swish, ov::op::v4)

// New operations added in opset5
_OPENVINO_OP_REG(GRUSequence, ov::op::v5)
_OPENVINO_OP_REG(HSigmoid, ov::op::v5)
_OPENVINO_OP_REG(LogSoftmax, ov::op::v5)
_OPENVINO_OP_REG(Loop, ov::op::v5)
_OPENVINO_OP_REG(LSTMSequence, ov::op::v5)
_OPENVINO_OP_REG(RNNSequence, ov::op::v5)
_OPENVINO_OP_REG(Round, ov::op::v5)

// New operations added in opset6
_OPENVINO_OP_REG(CTCGreedyDecoderSeqLen, ov::op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronDetectionOutput, ov::op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronGenerateProposalsSingleImage, ov::op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronPriorGridGenerator, ov::op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronROIFeatureExtractor, ov::op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronTopKROIs, ov::op::v6)
_OPENVINO_OP_REG(GatherElements, ov::op::v6)
_OPENVINO_OP_REG(MVN, ov::op::v6)
_OPENVINO_OP_REG(Assign, ov::op::v6)     // new version
_OPENVINO_OP_REG(ReadValue, ov::op::v6)  // new version

// New operations added in opset7
_OPENVINO_OP_REG(DFT, ov::op::v7)
_OPENVINO_OP_REG(Einsum, ov::op::v7)
_OPENVINO_OP_REG(Gelu, ov::op::v7)
_OPENVINO_OP_REG(IDFT, ov::op::v7)
_OPENVINO_OP_REG(Roll, ov::op::v7)

// New operations added in opset8
_OPENVINO_OP_REG(Gather, ov::op::v8)
_OPENVINO_OP_REG(GatherND, ov::op::v8)
_OPENVINO_OP_REG(AdaptiveAvgPool, ov::op::v8)
_OPENVINO_OP_REG(AdaptiveMaxPool, ov::op::v8)
_OPENVINO_OP_REG(DeformableConvolution, ov::op::v8)
_OPENVINO_OP_REG(DetectionOutput, ov::op::v8)
_OPENVINO_OP_REG(I420toBGR, ov::op::v8)
_OPENVINO_OP_REG(I420toRGB, ov::op::v8)
_OPENVINO_OP_REG(MatrixNms, ov::op::v8)
_OPENVINO_OP_REG(MaxPool, ov::op::v8)
_OPENVINO_OP_REG(NV12toBGR, ov::op::v8)
_OPENVINO_OP_REG(NV12toRGB, ov::op::v8)
_OPENVINO_OP_REG(RandomUniform, ov::op::v8)
_OPENVINO_OP_REG(Slice, ov::op::v8)
_OPENVINO_OP_REG(Softmax, ov::op::v8)
_OPENVINO_OP_REG(If, ov::op::v8)
_OPENVINO_OP_REG(PriorBox, ov::op::v8)

// New operations added in opset9
_OPENVINO_OP_REG(IRDFT, ov::op::v9)
_OPENVINO_OP_REG(RDFT, ov::op::v9)
_OPENVINO_OP_REG(Eye, ov::op::v9)
_OPENVINO_OP_REG(NonMaxSuppression, ov::op::v9)
_OPENVINO_OP_REG(ROIAlign, ov::op::v9)
_OPENVINO_OP_REG(SoftSign, ov::op::v9)
_OPENVINO_OP_REG(GenerateProposals, ov::op::v9)
_OPENVINO_OP_REG(MulticlassNms, ov::op::v9)

// New operations added in opset10
_OPENVINO_OP_REG(IsFinite, ov::op::v10)
_OPENVINO_OP_REG(IsInf, ov::op::v10)
_OPENVINO_OP_REG(IsNaN, ov::op::v10)
_OPENVINO_OP_REG(Unique, ov::op::v10)

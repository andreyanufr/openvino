// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <low_precision/gather.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "layer_transformation.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/gather_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {
using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;

class GatherTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    std::vector<size_t> gatherIndicesShape;
    std::vector<int> gatherIndicesValues;
    std::vector<int> axis;
    int64_t batch_dims;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<ngraph::PartialShape, GatherTransformationTestValues, int> GatherTransformationParams;

class GatherTransformation : public LayerTransformation,
                             public testing::WithParamInterface<GatherTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const GatherTransformationTestValues testValues = std::get<1>(GetParam());
        const int opset_version = std::get<2>(GetParam());

        actualFunction =
            ngraph::builder::subgraph::GatherFunction::getOriginal(inputShape,
                                                                   testValues.gatherIndicesShape,
                                                                   testValues.gatherIndicesValues,
                                                                   testValues.axis,
                                                                   testValues.batch_dims,
                                                                   testValues.actual.precisionBeforeDequantization,
                                                                   testValues.actual.dequantization,
                                                                   opset_version);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::GatherTransformation, ngraph::opset1::Gather>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction =
            ngraph::builder::subgraph::GatherFunction::getReference(inputShape,
                                                                    testValues.gatherIndicesShape,
                                                                    testValues.gatherIndicesValues,
                                                                    testValues.axis,
                                                                    testValues.batch_dims,
                                                                    testValues.expected.precisionBeforeDequantization,
                                                                    testValues.expected.dequantizationBefore,
                                                                    testValues.expected.precisionAfterOperation,
                                                                    testValues.expected.dequantizationAfter,
                                                                    opset_version);
    }

    static std::string getTestCaseName(testing::TestParamInfo<GatherTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const GatherTransformationTestValues testValues = std::get<1>(obj.param);
        const int opset_version = std::get<2>(obj.param);

        std::ostringstream result;
        result << "_" << inputShape << "_" << testValues.gatherIndicesShape << "_" << testValues.gatherIndicesValues
               << "_" << testValues.axis << "_" << testValues.batch_dims << "_"
               << testValues.actual.precisionBeforeDequantization << "_" << testValues.actual.dequantization << "_"
               << testValues.expected.dequantizationBefore << "_" << opset_version;
        return result.str();
    }
};

TEST_P(GatherTransformation, CompareFunctions) {
    ov::pass::InitNodeInfo().run_on_model(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<int> opset_version = {1, 7, 8};

const std::vector<ngraph::PartialShape> inputShapes3D = {{3, 3, 4}, {-1, -1, -1}};

const std::vector<GatherTransformationTestValues> testValues = {
    // U8: per-tensor quantization
    {{1},
     {0},
     {0},
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32}, {{128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true}, {0.1f}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32}, {{128}, ngraph::element::f32, {}, true, 1, ngraph::element::u8, true}, {0.1f}}}},
    // U8: per-tensor quantization
    {{2},
     {0, 1},
     {0},
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}},
     {ngraph::element::u8, {{}, {}, {}}, ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}}},
    // U8: per-tensor quantization
    {{3, 2},
     {1, 2, 1, 2, 1, 2},
     {1},
     std::int64_t{1},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}},
     {ngraph::element::u8, {{}, {}, {}}, ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}}},
    // U8: per-channel quantization with the same values
    {{1},
     {0},
     {0},
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128.f}, element::undefined, {1, 3, 1}, false, 1ul, element::u8, true},
       {{0.1f}, ngraph::element::f32, {1, 3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32},
       {{128.f}, element::undefined, {1, 3, 1}, false, 1ul, element::u8, true},
       {{0.1f}, ngraph::element::f32, {1, 3, 1}}}}},
    // U8: per-channel quantization, gather axis match with channel
    {{1},
     {0},
     {1},  // axis
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {1, 3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {1, 3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32}, {{128}, ngraph::element::f32, {}}, {{0.3f}, ngraph::element::f32, {}}}}},
    // U8: per-channel quantization, gather axis match with channel, quantization constant shape size is
    // less than input shape
    {{1},
     {1},
     {1},  // axis
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32}, {{64}, ngraph::element::f32, {}}, {{0.2f}, ngraph::element::f32, {}}}}},
    // U8: per-channel quantization, gather axis and channel doesn't match
    {{1},
     {0},
     {0},
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {1, 3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {1, 3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {1, 3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {1, 3, 1}}}}},
    // U8: per-channel quantization,  negative axis, gather axis match with channel
    {{1},
     {0},
     {-2},  // axis
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {1, 3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {1, 3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32}, {{128}, ngraph::element::f32, {}}, {{0.3f}, ngraph::element::f32, {}}}}},
    // empty
    {{1},
     {0},
     {0},
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8, {}},
     {ngraph::element::u8, {}, ngraph::element::u8, {}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         GatherTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn(opset_version)),
                         GatherTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<int> opset_version = {8};

const std::vector<ngraph::PartialShape> inputShapes3D = {{3, 3, 4}, {-1, -1, -1}};

const std::vector<GatherTransformationTestValues> testValues = {
    // U8: per-tensor quantization, negative indices value
    {{3, 2},
     {-2, 2, -2, 2, -2, 2},  // indices value
     {1},
     std::int64_t{1},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}},
     {ngraph::element::u8, {{}, {}, {}}, ngraph::element::u8, {{ngraph::element::f32}, {128}, {0.1f}}}},
    // U8: per-channel quantization, negative indices value, gather axis match with channel
    {{1},
     {-1},  // indices value
     {1},   // axis
     std::int64_t{0},
     LayerTransformation::createParamsU8I8(),
     {ngraph::element::u8,
      {{ngraph::element::f32},
       {{128, 64, 32}, ngraph::element::f32, {1, 3, 1}},
       {{0.3f, 0.2f, 0.1f}, ngraph::element::f32, {1, 3, 1}}}},
     {ngraph::element::u8,
      {{}, {}, {}},
      ngraph::element::u8,
      {{ngraph::element::f32}, {{32}, ngraph::element::f32, {}}, {{0.1f}, ngraph::element::f32, {}}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         GatherTransformation,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::ValuesIn(testValues),
                                            ::testing::ValuesIn(opset_version)),
                         GatherTransformation::getTestCaseName);
}  // namespace testValues2

}  // namespace

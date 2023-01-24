// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include <subgraph_customizable.hpp>
#include <snippets_helpers.hpp>
#include <ngraph_transformations/snippets_mark_skipped.hpp>
#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

class SnippetsMarkSkippedTests : public TransformationTestsF {
public:
    void run() {
        ASSERT_TRUE(function);
        manager.register_pass<ov::intel_cpu::SnippetsMarkSkipped>();
        manager.register_pass<ngraph::snippets::pass::EnumerateNodes>();
        manager.register_pass<ngraph::snippets::pass::TokenizeSnippets>();
        //
        // todo: This is a temporary work-around. remove when MatMul tokenization is supported through general pipeline
        manager.get_pass_config()->set_callback<ngraph::snippets::pass::TokenizeSnippets>(
                [](const std::shared_ptr<const ov::Node>& n) -> bool {
                        return ov::is_type<const ov::op::v0::MatMul>(n);
                });
    }
};

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipAfterInputsMatMulEltwise) {
    const auto &f = MatMulEltwiseBranchesFunction(std::vector<PartialShape> {{1, 3, 4, 4}, {1, 3, 4, 4}});
    function = f.getOriginal();
    // Fully tokenizable, since inputs are followed by MatMul
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_Snippets_SkipConvFused_ConvMulActivation) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Multiply>(),
                                                   std::make_shared<ov::op::v0::Tanh>(),
                                                   std::make_shared<ov::op::v0::Sqrt>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    function = f.getOriginal();
    // Fully tokenizable, since Mul with 2 inputs isn't fused into Convolution
    function_ref = f.getReference();
    run();
}

TEST_F(SnippetsMarkSkippedTests, smoke_SkipConvFused_ConvSumActivation) {
    std::vector<std::shared_ptr<Node>> eltwiseOps {std::make_shared<ov::op::v1::Add>(),
                                                   std::make_shared<ov::op::v0::Tanh>(),
                                                   std::make_shared<ov::op::v0::Sqrt>()};
    std::vector<PartialShape> inputShapes {{1, 2, 16, 16}, {1, 2, 1, 16}};
    const auto &f = ConvMulActivationFunction(inputShapes, eltwiseOps);
    function = f.getOriginal();
    // Not tokenizable, since Add + Eltwises can be fused into Convolution
    function_ref = f.getOriginal();
    run();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
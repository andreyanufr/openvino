// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/elementwise_args.hpp"

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "utils.hpp"

std::tuple<ov::element::Type, ov::PartialShape> ov::op::util::validate_and_infer_elementwise_args(Node* node) {
    OPENVINO_ASSERT(node != nullptr, "Node is empty! Cannot validate eltwise arguments.");
    constexpr size_t valid_inputs_count = 2;
    NODE_VALIDATION_CHECK(node,
                          node->get_input_size() == valid_inputs_count,
                          "Incorrect number of inputs. Required: ",
                          valid_inputs_count);

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        node,
        element::Type::merge(result_et, node->get_input_element_type(0), node->get_input_element_type(1)),
        "Arguments do not have the same element type (arg0 element type: ",
        node->get_input_element_type(0),
        ", arg1 element type: ",
        node->get_input_element_type(1),
        ").");

    const auto& A_shape = node->get_input_partial_shape(0);
    const auto& B_shape = node->get_input_partial_shape(1);
    std::vector<ov::PartialShape> input_shapes = {A_shape, B_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    eltwise_shape_infer(node, input_shapes, output_shapes);

    return std::make_tuple(result_et, output_shapes[0]);
}

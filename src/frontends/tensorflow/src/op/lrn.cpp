// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_lrn_op(const NodeContext& node) {
    // The normalization is performed along the last dimension
    default_op_checks(node, 1, {"LRN"});
    auto input = node.get_input(0);

    // retrieve attributes
    auto depth_radius = node.get_attribute<int64_t>("depth_radius", 5);
    auto bias = static_cast<double>(node.get_attribute<float>("bias", 1));
    auto alpha = static_cast<double>(node.get_attribute<float>("alpha", 1));
    auto beta = static_cast<double>(node.get_attribute<float>("beta", 0.5));

    // adjust attribute values for opset LRN operation
    size_t attr_size = static_cast<size_t>(depth_radius) * 2 + 1;
    alpha *= static_cast<double>(attr_size);

    // currently, plugins fallback to the template to execute this operation
    // the current implementation supports only the normalization across channel (axes={1})
    // and spatial dimensions (axes={2,3})
    // so we have to transpose inputs and outputs due to this limitation
    auto axis = make_shared<Constant>(element::i32, Shape{1}, 1);
    ov::frontend::tensorflow::convert_nhwc_to_nchw(true, input);
    auto lrn = make_shared<LRN>(input, axis, alpha, beta, bias, attr_size)->output(0);
    ov::frontend::tensorflow::convert_nchw_to_nhwc(true, lrn);
    set_node_name(node.get_name(), lrn.get_node_shared_ptr());
    return {lrn};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

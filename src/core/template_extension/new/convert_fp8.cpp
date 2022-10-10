// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fp8.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ConvertFP8::ConvertFP8(const ov::Output<ov::Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void ConvertFP8::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> ConvertFP8::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<ConvertFP8>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool ConvertFP8::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool ConvertFP8::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto in = inputs[0];
    auto out = outputs[0];
    if (out.data() == in.data())  // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool ConvertFP8::has_evaluate() const {
    return true;
}
//! [op:evaluate]

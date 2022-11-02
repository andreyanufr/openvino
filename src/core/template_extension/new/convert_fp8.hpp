// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
#include <openvino/op/convert.hpp>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

enum class TypeFP8 {
    bf8, //!< 1s5e2m element type
    hf8  //!< 1s4e3m element type
};


class ConvertFP8 : public ov::op::Op {
public:
    OPENVINO_OP("ConvertFP8");

    ConvertFP8() = default;
    ConvertFP8(const ov::Output<ov::Node>& arg,
               const ov::Output<ov::Node>& input_low,
               const std::string& destination_type,
               bool is_weight = false);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    void validate() const;
    std::string m_destination_type = "hf8";
    bool m_is_weight = false;
};
//! [op:header]

}  // namespace TemplateExtension

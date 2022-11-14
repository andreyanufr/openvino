// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
#include <openvino/op/convert.hpp>
//! [op:common_include]

#pragma once

//! [op:header]
namespace ov {
namespace op {
namespace v1 {
    class ConvertFP8 : public Op {
    public:
        OPENVINO_OP("ConvertFP8", "opset1", op::Op, 1);

        ConvertFP8();
        ConvertFP8(const ov::Output<ov::Node>& arg,
                   const ov::Output<ov::Node>& scale,
                   const std::string& destination_type);

        void validate_and_infer_types() override;
        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
        bool visit_attributes(ov::AttributeVisitor& visitor) override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
        bool has_evaluate() const override;

    private:
        void validate() const;
        std::string m_destination_type = "hf8_ext";
        static const std::vector<std::string> m_valid_types;
    };
    //! [op:header]

}  // namespace v1
}  // namespace op
}  // namespace ov

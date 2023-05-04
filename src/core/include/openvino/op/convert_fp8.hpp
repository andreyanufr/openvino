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
namespace v10 {
    class OPENVINO_API FakeConvertFP : public Op {
    public:
        OPENVINO_OP("FakeConvertFP", "opset10");
        BWDCMP_RTTI_DECLARATION;

        FakeConvertFP();
        FakeConvertFP(const ov::Output<ov::Node>& arg,
                   const ov::Output<ov::Node>& scale,
                   const ov::Output<ov::Node>& shift,
                   const std::string& destination_type,
                   bool apply_scale);

        void validate_and_infer_types() override;
        std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
        bool visit_attributes(ov::AttributeVisitor& visitor) override;

        bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
        bool has_evaluate() const override;

    private:
        void validate() const;
        std::string m_destination_type = "HF8";
        bool m_apply_scale = false;
        static const std::vector<std::string> m_valid_types;
    };
    //! [op:header]

}  // namespace v1
}  // namespace op
}  // namespace ov

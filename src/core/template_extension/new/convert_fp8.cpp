// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fp8.hpp"

#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ConvertFP8::ConvertFP8(const ov::Output<ov::Node>& arg, TypeFP8& destination_type) : 
    Op({arg}),
    m_destination_type(destination_type),
    m_convert_fp16(std::make_shared < ov::op::v0::Convert>(arg, ov::element::f16))
{
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void ConvertFP8::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> ConvertFP8::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<ConvertFP8>(new_args.at(0), m_destination_type);
}
//! [op:copy]

//! [op:visit_attributes]
bool ConvertFP8::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("destination_type", m_destination_type);
}
//! [op:visit_attributes]

//! [op:evaluate]
bool ConvertFP8::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (m_convert_fp16.get() == nullptr)
        return false;
    ov::TensorVector fp16;

    fp16.emplace_back(
        std::make_shared<ngraph::runtime::HostTensor>(ov::element::f16, inputs[0].get_shape()));

    m_convert_fp16->evaluate(fp16, inputs);
    return true;
}

/// <summary>
/// emulation of convertation fp16 value to bf8 1s5e2m format 
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
//template <typename T>
void convertfp16_bf8(const unsigned short* arg,
                     unsigned short* out,
                     size_t count,
                     int exp_bits = 6,
                     int mbits = 10) {
    typedef union half_t {
        unsigned short u;
        unsigned short f;
    } __half_t;

    int non_mant_bits = exp_bits + 1;           /* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);  // 10 - (10 - 6) == 6 ???

    unsigned short mask_mant = (unsigned short)(0xffff << lshift);  // 1111111111111111 -> 1111111111000000
    unsigned short grs_bitmask = 0x00ff;                            // 0 00000 0011111111
    unsigned short rne_tie = 0x0180;                                // 0 00000 0110000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        h.f = arg[i];
        unsigned short is_normal = 1;
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        is_normal = (((h.u & 0x7c00) <= 0x7800) && ((h.u & 0x7c00) >= 0x0400)) ? 1 : 0;
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        unsigned short can_round = ((h.u & 0x7f00) < 0x7b00) ? 1 : 0;
        /// denormalized numbers including zero, all exponent valuse equal to zero
        unsigned short is_denorm = ((h.u & 0x7c00) == 0x0) ? 1 : 0;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        unsigned short is_naninf = ((h.u & 0x7c00) == 0x7c00) ? 1 : 0;
        /* nearest rounding masks */
        /// grs_bitmask - 0x00ff is 0 00000 0011111111 or 255.0 - lower mantissa bits outside of bf8
        unsigned short rnmask = (h.u & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0 ???
        unsigned short rnmask_tie = (h.u & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            h.u += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        h.u = (h.u & mask_mant); /* truncation */
        out[i] = h.f;
    }
}

/// <summary>
/// emulation of convertation fp16 value to bf8 1s5e2m format
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// template <typename T>
void convertfp16_bf8(const unsigned short* arg, unsigned short* out, size_t count, int exp_bits = 6, int mbits = 10) {
    typedef union half_t {
        unsigned short u;
        unsigned short f;
    } __half_t;

    int non_mant_bits = exp_bits + 1;           /* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);  // 10 - (10 - 6) == 6 ???

    unsigned short mask_mant = (unsigned short)(0xffff << lshift);  // 1111111111111111 -> 1111111111000000
    unsigned short grs_bitmask = 0x00ff;                            // 0 00000 0011111111
    unsigned short rne_tie = 0x0180;                                // 0 00000 0110000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        h.f = arg[i];
        unsigned short is_normal = 1;
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        is_normal = (((h.u & 0x7c00) <= 0x7800) && ((h.u & 0x7c00) >= 0x0400)) ? 1 : 0;
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        unsigned short can_round = ((h.u & 0x7f00) < 0x7b00) ? 1 : 0;
        /// denormalized numbers including zero, all exponent valuse equal to zero
        unsigned short is_denorm = ((h.u & 0x7c00) == 0x0) ? 1 : 0;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        unsigned short is_naninf = ((h.u & 0x7c00) == 0x7c00) ? 1 : 0;
        /* nearest rounding masks */
        /// grs_bitmask - 0x00ff is 0 00000 0011111111 or 255.0 - lower mantissa bits outside of bf8
        unsigned short rnmask = (h.u & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0 ???
        unsigned short rnmask_tie = (h.u & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            h.u += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        h.u = (h.u & mask_mant); /* truncation */
        out[i] = h.f;
    }
}

bool ConvertFP8::has_evaluate() const {
    return true;
}
//! [op:evaluate]


namespace convert {
namespace {
template <ov::element::Type_t INPUT_ET, ov::element::Type_t OUTPUT_ET>
bool evaluate(const ov::HostTensorPtr& arg, const ov::HostTensorPtr& out)

{
    out->set_shape(arg->get_shape());
    size_t element_count = shape_size(out->get_shape());

    if ((INPUT_ET != arg->get_element_type()) || OUTPUT_ET != out->get_element_type()) {
        return false;
    }
    if (((INPUT_ET == element::u1) || (OUTPUT_ET == element::u1)) ||
        ((INPUT_ET == element::u4) || (OUTPUT_ET == element::u4)) ||
        ((INPUT_ET == element::i4) || (OUTPUT_ET == element::i4))) {
        runtime::reference::detail::lp_convert(arg->get_data_ptr<INPUT_ET>(),
                                               out->get_data_ptr<OUTPUT_ET>(),
                                               element_count,
                                               INPUT_ET,
                                               OUTPUT_ET);
    } else {
        runtime::reference::convert(arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count);
    }
    return true;
}
}  // namespace
}  // namespace convert

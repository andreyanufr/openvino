// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fp8.hpp"

#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ConvertFP8::ConvertFP8(const ov::Output<ov::Node>& arg, const ov::element::Type& destination_type)
    : 
    Op({arg}),
    m_destination_type(destination_type),
    m_convert_fp16(std::make_shared<ov::op::v0::Convert>(arg, ov::element::f16)),
    m_convert_fp32(std::make_shared<ov::op::v0::Convert>(arg, ov::element::f32)) {
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
    validate();
    visitor.on_attribute("destination_type", m_destination_type);

    return true;
}
//! [op:visit_attributes]

void ConvertFP8::validate() const {
    OPENVINO_ASSERT(m_destination_type == ov::element::bf8 || m_destination_type == ov::element::hf8,
                    "Bad format for f8 conversion type. Allowed types: [ov::element::bf8, ov::element::hf8]");
}

bool ConvertFP8::has_evaluate() const {
    return true;
}



namespace convert_fp8 {
namespace {
void print_tensor(const ov::Tensor& t, std::string s) {
    std::cout << "Tensor " << s << ": ";
    auto shape = t.get_shape();
    int len = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
    if (t.get_element_type() == ov::element::f16) {
        auto ptr = static_cast<ov::float16*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    if (t.get_element_type() == ov::element::f32) {
        auto ptr = static_cast<float*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    std::cout << std::endl;
}
/// <summary>
/// emulation of convertation fp16 value to bf8 1s-5e-2m format, Brain Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
template <typename T>
void convertfp16_bf8(const T* const arg, T* out, size_t count, int exp_bits = 6, int mbits = 8) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    int non_mant_bits = exp_bits + 1;           /* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);  // 10 - (10 - 6) == 6 ???

    unsigned short mask_mant = (unsigned short)(0xffff << lshift);  // 1111111111111111 -> 1 11111 1111000000
    unsigned short grs_bitmask = 0x00ff;                            // 0 00000 0011111111 - guard, round, sticky bits
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
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        unsigned short rnmask = (h.u & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0 ???
        unsigned short rnmask_tie = (h.u & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            h.u += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        h.u = (h.u & mask_mant); /* truncation */
        out[i] = h.f;
    }
}

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Hybrid Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// Exponent denormal values 0 -11
// Exponent normal values 1 -10
// Exponent normal values 2 -9
// Exponent normal values 3 -8
// Exponent normal values 4 -7
// Exponent normal values 5 -6
// Exponent normal values 6 -5
// Exponent normal values 7 -4
// Exponent normal values 8 -3
// Exponent normal values 9 -2
// Exponent normal values 10 -1
// Exponent normal values 11 0
// Exponent normal values 12 1
// Exponent normal values 13 2
// Exponent normal values 14 3
// Exponent NaN values 15 4
template <typename T>
void convertfp16_hf8(const T* arg, T* out, size_t count, int exp_bits = 6, int mbits = 9) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    //runtime::reference::clamp<T>(arg->get_data_ptr<T>(), arg->get_data_ptr<T>(), 15.0, 15.0, count);

    int non_mant_bits = exp_bits + 1; /* exponent + sign */         ///  6 - ?
    int lshift = 10 - (mbits - non_mant_bits);                      /// 10 - (9 - 6) == 7 - ???
    unsigned short rne_mask = 1;                                    /* round to nearest even mask */
    unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);  // 1111111111111111 -> 1 11111 1111000000
    unsigned short grs_bitmask = 0x007F;                            /// 0 00000 0001111111
    unsigned short rne_tie = 0x00C0;                                /// 0 00000 0011000000

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        h.f = arg[i];
        float inval = ngraph::float16(arg[i]);
        /* flush values below 1-4-3 (offset=4) subnormal range to zero */
        if (fabs(inval) < 1.2207031e-4)
            h.f = 0;

        short exp_h = (short)((h.u & 0x7C00) >> 10) - 15;  /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent
        short sign_h = (h.u & 0x8000);           /// 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);       /// 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326) - ????
        unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
        unsigned short is_normal = 1;

        is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
        unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 3) {  // too large, set it to NaN or inf
            mantissa_h = 0;
            exp_h = 16;
            is_naninf = 1;
        } else if (exp_h < -13) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }
        /* nearest rounding masks, & 0 00000 0001111111 - mantissa bits below hf8 */
        unsigned short rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between hf8 and fp16 mantissa */
        unsigned short rnmask_tie = (mantissa_h & rne_tie);
        if (!is_naninf && can_round && rne_mask) {
            /* round to nearest even, if rne_mask is enabled */
            /// rnmask > 0 00000 0001000000(64) or 0 00000 0011000000 - edge bits is 1
            /// += 0 00000 0010000000
            mantissa_h += (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        }
        if (exp_h < -10) { /* handle denormals -13, -12, -11, dshift 1, 2, 3 */
            dshift = (-10 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
    }
}

template <typename ET>
bool evaluate(const ov::Tensor& arg, ov::Tensor& out, const ov::element::Type& destination_type) {
    /*out->set_shape(arg->get_shape());
    size_t element_count = shape_size(out->get_shape());

    if ((INPUT_ET != arg->get_element_type()) || OUTPUT_ET != out->get_element_type()) {
        return false;
    }

    if (destination_type == ov::element::bf8)
        return convertfp16_bf8(arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count);
    else if (destination_type == ov::element::hf8)
        return convertfp16_hf8(arg->get_data_ptr<INPUT_ET>(), out->get_data_ptr<OUTPUT_ET>(), element_count);*/

    out.set_shape(arg.get_shape());
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    if (destination_type == ov::element::bf8)
        convertfp16_bf8(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    else if (destination_type == ov::element::hf8)
        convertfp16_hf8(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    else {
        std::cout << "Bad destination_type: " << destination_type << std::endl;
    }

    return true;
}

}  // namespace
}  // namespace convert


//! [op:evaluate]
bool ConvertFP8::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (m_convert_fp16.get() == nullptr)
        return false;
    validate();

    ov::TensorVector fp16;

    convert_fp8::print_tensor(inputs[0], "inputs");

    fp16.emplace_back(ov::Tensor(ov::element::f16, inputs[0].get_shape()));

    m_convert_fp16->evaluate(fp16, inputs);

    convert_fp8::print_tensor(fp16[0], "fp16");

    if (outputs[0].get_element_type() == ov::element::f16)
        convert_fp8::evaluate<unsigned short>(fp16[0], outputs[0], m_destination_type);
    else if (outputs[0].get_element_type() == ov::element::f32) {
        convert_fp8::evaluate<unsigned short>(fp16[0], fp16[0], m_destination_type);
        m_convert_fp32->evaluate(outputs, fp16);
    }

    convert_fp8::print_tensor(outputs[0], "outputs");

    return true;
}
//! [op:evaluate]
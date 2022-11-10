// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_fp8.hpp"

#include "ngraph/op/equal.hpp"
#include "ngraph/op/select.hpp"

using namespace TemplateExtension;

//! [op:ctor]
ConvertFP8::ConvertFP8(const ov::Output<ov::Node>& arg,
                       const ov::Output<ov::Node>& input_low,
                       const std::string& destination_type,
                       bool is_weight)
    : 
    Op({arg, input_low}),
    m_destination_type(destination_type),
    m_is_weight(is_weight)
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
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");

    return std::make_shared<ConvertFP8>(new_args.at(0),
                                        new_args.at(1),
                                        m_destination_type, m_is_weight);
}
//! [op:copy]

//! [op:visit_attributes]
bool ConvertFP8::visit_attributes(ov::AttributeVisitor& visitor) {
    //validate();
    visitor.on_attribute("destination_type", m_destination_type);
    visitor.on_attribute("is_weight", m_is_weight);

    return true;
}
//! [op:visit_attributes]

void ConvertFP8::validate() const {
    OPENVINO_ASSERT(m_destination_type == "bf8" || m_destination_type == "hf8" || m_destination_type == "hf8_eb_7" ||
                    m_destination_type == "hf8_eb_12",
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
    int len = shape_size(shape);

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
void convertfp16_bf8(const T* const arg, T* out, size_t count, int exp_bits = 5, int mbits = 8) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    int non_mant_bits = exp_bits + 1;           /* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);  // 10 - (8 - 6) == 8 ???

    unsigned short mask_mant = (unsigned short)(0xffff << lshift);  // 1111111111111111 -> 1 11111 1100000000
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
// Exponent normal values 1..14 -10..3 (11 - exponent)
// Exponent NaN values 15 4
template <typename T>
void convertfp16_hf8(const T* arg, T* out, size_t count, int exp_bits = 5,
                     int mbits = 9, bool use_clamp=true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;


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
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 3) {  // too large, set it to NaN or inf
            if (use_clamp) {
                exp_h = 3;
                mantissa_h = 0b0000001110000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
                is_naninf = 1;
            }
        } else if (exp_h < -13) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below hf8 (grs) */
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


/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Hybrid Float
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
// Exponent denormal values 0 -11
// Exponent normal values 1..14 -10..3 (11 - exponent)
// Exponent NaN values 15 4
template <typename T>
void convertfp16_hf8_ext(const T* arg, T* out, size_t count, int exp_bits = 5, int mbits = 9, bool use_clamp = true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

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

        short exp_h =
            (short)((h.u & 0x7C00) >> 10) - 15;  /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent
        short sign_h = (h.u & 0x8000);           /// 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);       /// 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326) - ????
        unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
        unsigned short is_normal = 1;

        is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 4) {  // too large, set it to NaN or inf
            if (use_clamp) {
                exp_h = 4;
                mantissa_h = 0b0000001100000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
                is_naninf = 1;
            }
        } else if (exp_h < -13) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }

        if (exp_h == 4 && mantissa_h > 0b0000001100000000) {
            mantissa_h = 0b0000001100000000;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below hf8 (grs) */
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

template <typename T>
void convertfp16_hf8_eb12(const T* arg, T* out, size_t count, int exp_bits = 5, int mbits = 9, bool use_clamp = true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

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
        //if (fabs(inval) < 0.5 * 1.2207031e-4)
        //    h.f = 0;
        if ((h.u & 0b0111111111111111) < 0b0000010000000000)
            h.f = 0;

        short exp_h =
            (short)((h.u & 0x7C00) >> 10) - 15;  /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent
        short sign_h = (h.u & 0x8000);           /// 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);       /// 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10001 1110000000 (7)
        unsigned short can_round = ((h.u & 0x7FFF) < 0b0100011110000000) ? 1 : 0;
        unsigned short is_normal = 1;

        is_normal = (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
        unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

        int dshift = 0;
        if (exp_h > 2) {  // too large, set it to NaN or inf
            if (use_clamp) {
                exp_h = 2;
                mantissa_h = 0b0000001110000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
                is_naninf = 1;
            }
        } else if (exp_h < -14) {  /// -14, -13, -12 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below hf8 (grs) */
        unsigned short rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between hf8 and fp16 mantissa */
        unsigned short rnmask_tie = (mantissa_h & rne_tie);
        if (!is_naninf && can_round && rne_mask) {
            /* round to nearest even, if rne_mask is enabled */
            /// rnmask > 0 00000 0001000000(64) or 0 00000 0011000000 - edge bits is 1
            /// += 0 00000 0010000000
            mantissa_h += (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        }
        if (exp_h < -11) { /* handle denormals -14, -13, -12, dshift 1, 2, 3 */
            dshift = (-11 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
    }
}

/// not original
#define LIBXSMM_CAST_USHORT(VALUE) \
    ((unsigned short)((VALUE)))


unsigned char convert_fp16_hf8(ov::float16 inp) {
    unsigned int f16_bias = 15;
    unsigned int f8_bias = 7;
    unsigned char res = 0;
    unsigned short s, e, m, e_f16, m_f16;
    unsigned int fixup;
    unsigned short in = inp.to_bits();

    s = (in & 0x8000) >> 8;
    e_f16 = (in & 0x7c00) >> 10;  /// & 0b0111110000000000
    m_f16 = (in & 0x03ff);        /// & 0b0000001111111111

    /* special value --> make it NaN */
    if (e_f16 == 0x1f) {  // == 31 or 0000000000011111
        e = 0xf; // 0000000000001111
        m = 0x7; // 0000000000000111
        /* overflow --> make it NaN */
    } else if ((e_f16 > (f16_bias - f8_bias + 15)) || ((e_f16 == (f16_bias - f8_bias + 15)) && (m_f16 > 0x0300))) { // exp >= 10111 ???
        e = 0xf;
        m = 0x7;
        /* smaller than denormal f8 + eps */
    } else if (e_f16 < f16_bias - f8_bias - 3) { // 5
        e = 0x0;
        m = 0x0;
        /* denormal */
    } else if (e_f16 <= f16_bias - f8_bias) {
        /* RNE */
        /* denormalized mantissa */
        m = m_f16 | 0x0400;
        /* addtionally subnormal shift */
        m = m >> ((f16_bias - f8_bias) + 1 - e_f16);
        /* preserve sticky bit (some sticky bits are lost when denormalizing) */
        m |= (((m_f16 & 0x007f) + 0x007f) >> 7);
        /* RNE Round */
        fixup = (m >> 7) & 0x1;
        m = m + LIBXSMM_CAST_USHORT(0x003f + fixup);
        m = m >> 7;
        e = 0x0;
        /* normal */
    } else {
        /* RNE round */
        fixup = (m_f16 >> 7) & 0x1;
        in = in + LIBXSMM_CAST_USHORT(0x003f + fixup);
        e = (in & 0x7c00) >> 10;
        m = (in & 0x03ff);
        OPENVINO_ASSERT(e >= LIBXSMM_CAST_USHORT(f16_bias - f8_bias), "");
        e -= LIBXSMM_CAST_USHORT(f16_bias - f8_bias);
        m = m >> 7;
    }

    /* set result to 0 */
    res = 0x0;
    /* set exp and mant */
    res |= e << 3;
    res |= m;
    /* sign it */
    res |= s;

    return res;
}

unsigned short convert_hf8_fp16(unsigned char inp) {
    unsigned int f16_bias = 15;
    unsigned int f8_bias = 7;
    unsigned short s = (inp & 0x80) << 8;
    unsigned short e = (inp & 0x78) >> 3;
    unsigned short m = (inp & 0x07);
    unsigned short e_norm = e + (f16_bias - f8_bias);
    unsigned short res = 0;
    /* convert denormal fp8 number into a normal fp16 number */
    if ((e == 0) && (m != 0)) {
        unsigned int lz_cnt = 2;
        lz_cnt = (m > 0x1) ? 1 : lz_cnt;
        lz_cnt = (m > 0x3) ? 0 : lz_cnt;
        OPENVINO_ASSERT(e_norm >= lz_cnt, "e_norm >= lz_cnt");
        e_norm -= lz_cnt;
        m = (m << (lz_cnt + 1)) & 0x07;
    } else if ((e == 0) && (m == 0)) {
        e_norm = 0;
    } else if ((e == 0xf) && (m == 0x7)) {
        e_norm = 0xff;
        m = 0x4; /* making first mantissa bit 1 */
    }

    /* set exp and mant */
    res |= (e_norm << 10);
    res |= (m << 7);
    /* sign it */
    res |= s;
    return res;
}

unsigned short convert_fp16_hf8_fp16(ov::float16 inp) {
    return convert_hf8_fp16(convert_fp16_hf8(inp));
}

/// <summary>
/// emulation of convertation fp16 value to hf8 1s-4e-3m format, Hybrid Float
/// exponent bias is 7
/// </summary>
/// <typeparam name="T">Every possible type with 16 bit size</typeparam>
/// <param name="arg"></param>
/// <param name="out"></param>
/// <param name="count"></param>
template <typename T>
void convertfp16_hf8_libxsmm(const T* arg,
                             T* out,
                             size_t count,
                             bool use_clamp = true) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        h.f = arg[i];
        h.u = convert_fp16_hf8_fp16(h.u);
        out[i] = h.f;
    }
}

template <typename ET>
bool evaluate(ov::Tensor& arg, ov::Tensor& out, const std::string& destination_type) {
    out.set_shape(arg.get_shape());
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    if (destination_type == "bf8") {
        convertfp16_bf8(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    } else if (destination_type == "hf8") {
        convertfp16_hf8(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    } else if (destination_type == "hf8_ext") {
        convertfp16_hf8_ext(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    } else if (destination_type == "hf8_eb_7") {
        convertfp16_hf8_libxsmm(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    } else if (destination_type == "hf8_eb_12") {
        convertfp16_hf8_eb12(static_cast<ET*>(arg.data()), static_cast<ET*>(out.data()), element_count);
    } else {
        std::cout << "Bad destination_type: " << destination_type << std::endl;
    }

    return true;
}


template <typename ET>
bool evaluate_mixed(ov::Tensor& arg, ov::Tensor& out, const ov::Tensor& scale) {
    out.set_shape(arg.get_shape());
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    if (ov::element::f32 != scale.get_element_type()) {
        std::cout << "Bad type of scale: " << scale.get_element_type() << std::endl;
        return false;
    }

    auto dataShape = arg.get_shape();
    auto scaleSize = scale.get_size();

    OPENVINO_ASSERT(dataShape[0] == scaleSize, "Shape mismatch in scale");

    size_t step = 1;
    for (size_t i = 1; i < dataShape.size(); i++) {
        step *= dataShape[i];
    }

    const float* scalePtr = static_cast<float*>(scale.data());
    for (size_t i = 0; i < scaleSize; i++) {
        auto inPtr = static_cast<ET*>(arg.data()) + i * step;
        auto outPtr = static_cast<ET*>(out.data()) + i * step;
        if (scalePtr[i] > 1.0) {
            convertfp16_bf8(intPtr, outPtr, step);
        } else {
            convertfp16_hf8_ext(intPtr, outPtr, step);
        }
    }

    return true;
}

template <typename T, typename S>
void apply_scale(T *data, int sz, S scale) {
    for (int i = 0; i < sz; i++) {
        data[i] = scale * data[i];
    }
}


template <typename T>
void apply_per_channel_scale(ov::Tensor& data, const ov::Tensor& scale, bool invert = false) {
    auto dataShape = data.get_shape();
    auto scaleSize = scale.get_size();

    T* dataPtr = static_cast<T*>(data.data());
    float* scalePtr = static_cast<float*>(scale.data());

    if (scaleSize == 1) { // per tensor scale, probably for activation 
        auto dataSize = data.get_size();
        float s = scalePtr[0];
        if (invert) {
            for (size_t j = 0; j < dataSize; j++) {
                dataPtr[j] /= s;
            }
        } else {
            for (size_t j = 0; j < dataSize; j++) {
                dataPtr[j] *= s;
            }
        }
        return;
    }

    OPENVINO_ASSERT(dataShape[0] == scaleSize, "Shape mismatch in scale");

    size_t step = 1;
    for (size_t i = 1; i < dataShape.size(); i++) {
        step *= dataShape[i];
    }

    for (size_t i = 0; i < scaleSize; i++) {
        T s = static_cast<T>(scalePtr[i]);
        if (invert) {
            for (size_t j = 0; j < step; j++) {
                dataPtr[j] /= s;
            }
        } else {
            for (size_t j = 0; j < step; j++) {
                dataPtr[j] *= s;
            }
        }
        dataPtr += step;
    }
}


void convert_to_fp16(const ov::Tensor& in, ov::Tensor& out) {
    auto inSz = in.get_size();
    auto outSz = out.get_size();

    OPENVINO_ASSERT(inSz == outSz, "Shape mismatch in scale");

    float* inPtr = static_cast<float*>(in.data());
    ov::float16* outPtr = static_cast<ov::float16*>(out.data());

    for (size_t i = 0; i < inSz; i++) {
        outPtr[i] = static_cast<ov::float16>(inPtr[i]);
    }
}


void convert_to_fp16_with_scale(const ov::Tensor& in, ov::Tensor& out, ov::Tensor& scale) {
    auto inSz = in.get_size();
    auto outSz = out.get_size();

    OPENVINO_ASSERT(inSz == outSz, "Shape mismatch in scale");

    float* inPtr = static_cast<float*>(in.data());
    ov::float16* outPtr = static_cast<ov::float16*>(out.data());

    for (size_t i = 0; i < inSz; i++) {
        outPtr[i] = static_cast<ov::float16>(inPtr[i]);
    }
}

void convert_to_fp32(const ov::Tensor& in, ov::Tensor& out) {
    auto inSz = in.get_size();
    auto outSz = out.get_size();

    OPENVINO_ASSERT(inSz == outSz, "Shape mismatch in scale");

    ov::float16* inPtr = static_cast<ov::float16*>(in.data());
    float*  outPtr = static_cast<float*>(out.data());

    for (size_t i = 0; i < inSz; i++) {
        outPtr[i] = static_cast<float>(inPtr[i]);
    }
}

}  // namespace
}  // namespace convert


//! [op:evaluate]
bool ConvertFP8::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    //validate();

    ov::TensorVector fp16;

    //convert_fp8::print_tensor(inputs[0], "inputs");

    fp16.emplace_back(ov::Tensor(ov::element::f16, inputs[0].get_shape()));

    //m_convert_fp16->evaluate(fp16, inputs);
    convert_fp8::convert_to_fp16(inputs[0], fp16[0]);

    if (outputs[0].get_element_type() == ov::element::f16) {
        if (m_is_weight) {
            convert_fp8::evaluate_mixed<unsigned short>(fp16[0], outputs[0], inputs[1]);
        } else {
            convert_fp8::evaluate<unsigned short>(fp16[0], outputs[0], m_destination_type);
        }
    }  else if (outputs[0].get_element_type() == ov::element::f32) {
        if (m_is_weight) {
            convert_fp8::evaluate_mixed<unsigned short>(fp16[0], fp16[0], inputs[1]);
        } else {
            convert_fp8::evaluate<unsigned short>(fp16[0], fp16[0], m_destination_type);
        }
        //m_convert_fp32->evaluate(outputs, fp16);
        convert_fp8::convert_to_fp32(fp16[0], outputs[0]);
    }

    //convert_fp8::print_tensor(outputs[0], "outputs");

    return true;
}
//! [op:evaluate]
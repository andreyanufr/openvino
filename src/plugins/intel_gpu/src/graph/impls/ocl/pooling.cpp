// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "pooling_inst.h"
#include "pooling/pooling_kernel_selector.h"
#include "pooling/pooling_kernel_base.h"
#include "ngraph/validation_util.hpp"

namespace cldnn {
namespace ocl {

namespace {

kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
    }
}

kernel_selector::kernel_divider_mode cldnn_2_kernel_divider_mode(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
        case pooling_mode::average:
            return kernel_selector::kernel_divider_mode::FIXED;
        case pooling_mode::average_no_padding:
            return kernel_selector::kernel_divider_mode::DYNAMIC;
        default:
            assert(0);
            return kernel_selector::kernel_divider_mode::DONT_CARE;
    }
}
}  // namespace

struct pooling_impl : typed_primitive_impl_ocl<pooling> {
    using parent = typed_primitive_impl_ocl<pooling>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::pooling_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::pooling_params, kernel_selector::pooling_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<pooling>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<pooling>();
        auto params = get_default_params<kernel_selector::pooling_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::pooling_optional_params>(impl_param.get_program());

        params.maxPoolOpset8Features = primitive->maxPoolOpset8Features;
        if (params.maxPoolOpset8Features) {
            switch (primitive->index_element_type) {
                case cldnn::data_types::i32: {
                    params.poolIndexElementType = kernel_selector::Datatype::INT32;
                    break;
                }
                case cldnn::data_types::i64: {
                    params.poolIndexElementType = kernel_selector::Datatype::INT64;
                    break;
                }
                default:
                    throw std::runtime_error{"Not supported index element type"};
            }
            params.poolAxis = primitive->axis;
        }

        const auto& input_layout = impl_param.get_input_layout();
        const auto& output_layout = impl_param.get_output_layout();

        auto kernel = primitive->size;
        auto stride = primitive->stride;
        auto dilation = primitive->dilation.empty() ? ov::Strides(stride.size(), 1)
                                                    : primitive->dilation;

        ov::CoordinateDiff pads_begin(primitive->pads_begin.begin(), primitive->pads_begin.end());
        ov::CoordinateDiff pads_end(primitive->pads_end.begin(), primitive->pads_end.end());
        auto auto_pad = primitive->auto_pad;

        if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
            pads_begin.clear();
            pads_end.clear();
            ngraph::try_apply_auto_padding(input_layout.get_partial_shape(),
                                           kernel,
                                           stride,
                                           dilation,
                                           auto_pad,
                                           pads_end,
                                           pads_begin);
        }
        if (auto_pad == ov::op::PadType::VALID) {
            pads_begin = ov::CoordinateDiff(pads_begin.size(), 0);
            pads_end = ov::CoordinateDiff(pads_end.size(), 0);
        }

        auto spatial_rank = output_layout.get_spatial_rank();

        kernel.resize(std::max<size_t>(2, kernel.size()), 1);
        stride.resize(std::max<size_t>(2, stride.size()), 1);
        dilation.resize(std::max<size_t>(2, dilation.size()), 1);
        pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
        pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);

        auto& pp = params;

        pp.poolType = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction = primitive->rounding_type == ov::op::RoundingType::CEIL ? kernel_selector::pool_remainder::CEIL
                                                                                    : kernel_selector::pool_remainder::FLOOR;

        // check if last pooling window goes outside of input size + padding. If so the avg pooling size will be
        // adjusted to that, to work properly this calculation must take pad_end into account.
        auto dynamic_mode = false;
        for (size_t i = 0; i < spatial_rank; i++) {
            dynamic_mode |= (((output_layout.spatial(i) - 1) * stride[spatial_rank - i - 1]) + primitive->size[spatial_rank - i - 1]) >
                                 static_cast<size_t>(pads_end[spatial_rank - i - 1] + pads_begin[spatial_rank - i - 1] + input_layout.spatial(i));
        }

        if (primitive->mode == pooling_mode::average && dynamic_mode)
            pp.divMode = kernel_selector::kernel_divider_mode::DYNAMIC_WITH_PADDING;
        else
            pp.divMode = cldnn_2_kernel_divider_mode(primitive->mode);

        uint32_t kernel_z = kernel.size() >= 3 ? static_cast<uint32_t>(kernel[kernel.size() - 3]) : 1;
        uint32_t kernel_y = kernel.size() >= 2 ? static_cast<uint32_t>(kernel[kernel.size() - 2]) : 1;
        uint32_t kernel_x = kernel.size() >= 1 ? static_cast<uint32_t>(kernel[kernel.size() - 1]) : 1;
        pp.poolSize = {kernel_x, kernel_y, kernel_z};

        uint32_t pad_z = std::max<std::ptrdiff_t>(pads_begin.size() >= 3 ? pads_begin[pads_begin.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pads_begin.size() >= 2 ? pads_begin[pads_begin.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pads_begin.size() >= 1 ? pads_begin[pads_begin.size() - 1] : 0, 0);
        pp.poolPad  = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? static_cast<uint32_t>(stride[stride.size() - 3]) : 1;
        uint32_t stride_y = stride.size() >= 2 ? static_cast<uint32_t>(stride[stride.size() - 2]) : 1;
        uint32_t stride_x = stride.size() >= 1 ? static_cast<uint32_t>(stride[stride.size() - 1]) : 1;
        pp.poolStride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? static_cast<uint32_t>(dilation[dilation.size() - 3]) : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? static_cast<uint32_t>(dilation[dilation.size() - 2]) : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? static_cast<uint32_t>(dilation[dilation.size() - 1]) : 1;
        pp.poolDilation = {dilation_x, dilation_y, dilation_z};

        return {params, optional_params};
    }
};

namespace detail {

attach_pooling_impl::attach_pooling_impl() {
    std::set<implementation_map<pooling>::key_type> keys;

    auto types = { data_types::f16, data_types::f32, data_types::i8, data_types::u8 };
    auto formats = { format::bfyx,
                     format::byxf,
                     format::yxfb,
                     format::b_fs_yx_fsv4,
                     format::b_fs_yx_fsv16,
                     format::b_fs_yx_fsv32,
                     format::bs_fs_yx_bsv16_fsv16,
                     format::bs_fs_yx_bsv16_fsv32,
                     format::bs_fs_yx_bsv32_fsv16,
                     format::bs_fs_yx_bsv32_fsv32,

                     format::bfzyx,
                     format::b_fs_zyx_fsv16,
                     format::b_fs_zyx_fsv32,
                     format::bs_fs_zyx_bsv16_fsv16,
                     format::bs_fs_zyx_bsv16_fsv32,
                     format::bs_fs_zyx_bsv32_fsv16,
                     format::bs_fs_zyx_bsv32_fsv32 };

    for (const auto type : types) {
        for (const auto format : formats) {
            keys.emplace(type, format);
        }
    }

    keys.emplace(data_types::f16, format::fs_b_yx_fsv32);
    keys.emplace(data_types::f32, format::fs_b_yx_fsv32);

    implementation_map<pooling>::add(impl_types::ocl, typed_primitive_impl_ocl<pooling>::create<pooling_impl>, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::pooling_impl)

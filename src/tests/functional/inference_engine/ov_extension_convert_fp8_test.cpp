// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <bitset>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"
#include "ie_iextension.h"
#include "ngraph/op/op.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/runtime/core.hpp"
#include "ngraph/runtime/reference/convert.hpp"


using namespace testing;
using namespace InferenceEngine;
using namespace CommonTestUtils;

class OVExtensionTests : public TestsCommon {
public:
    ov::Core core;

    void test() {
        std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ConvertFP8" version="extension" destination_type="bf16">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="out_data">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
        ov::Tensor weights;
        ov::PartialShape refBeforeReshape{1, 3, 22, 22};
        ov::PartialShape refAfterReshape{8, 9, 33, 66};

        auto network = core.read_model(model, weights);
        std::map<std::string, ov::PartialShape> newShapes;
        newShapes["in_data"] = refAfterReshape;

        EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
        EXPECT_NO_THROW(network->reshape(newShapes));
        EXPECT_EQ(refAfterReshape, network->output().get_partial_shape());
    }

    void replace_all(std::string& s, std::string const& toReplace, std::string const& replaceWith) {
        std::string buf;
        std::size_t pos = 0;
        std::size_t prevPos;

        // Reserves rough estimate of final size of string.
        buf.reserve(s.size());

        while (true) {
            prevPos = pos;
            pos = s.find(toReplace, pos);
            if (pos == std::string::npos)
                break;
            buf.append(s, prevPos, pos - prevPos);
            buf += replaceWith;
            pos += toReplace.size();
        }

        buf.append(s, prevPos, s.size() - prevPos);
        s.swap(buf);
    }

    void cmp_mat(const ov::Tensor &t1, const ov::Tensor &t2) {
        auto ptr1 = static_cast<uint16_t*>(t1.data());
        auto ptr2 = static_cast<uint16_t*>(t2.data());
        auto shape = t1.get_shape();
        int len = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());

        for (int i = 0; i < len; i++) {
            EXPECT_EQ(ptr1[i], ptr2[i]);
            if (ptr1[i] != ptr2[i]) {
                std::bitset<16> exp_val(ptr1[i]);
                std::bitset<16> out_val(ptr2[i]);
                std::cout << i << ") " <<  "Expected: " << exp_val << ". Received: " << out_val << std::endl;
            }
        }
    }

    void test_load_ir() {
        std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,<shape>,1,1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ConvertFP8" version="extension" destination_type="bf16">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="out_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

        replace_all(model, "<shape>", std::to_string(5));
        ov::Tensor weights;
        ov::PartialShape refBeforeReshape{1, 5, 1, 1};

        auto network = core.read_model(model, weights);

        EXPECT_EQ(refBeforeReshape, network->output().get_partial_shape());
    }

    void print_tensor(const ov::Tensor& t) {
        std::cout << "Tensor: ";
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
    /// 1s-5e-2m. The same size of exponent like in half precision.
    /// </summary>
    void test_bf8() {
                std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,<shape>,1,1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ConvertFP8" version="extension" destination_type="bf8">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="out_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

        ov::Tensor weights;
        const int sz = 9;
        replace_all(model, "<shape>", std::to_string(sz));
        std::string deviceName = "CPU";
        auto network = core.read_model(model, weights);

        ov::Shape shape{1, sz, 1, 1};
        ov::Tensor data(ov::element::f16, shape);
        ov::Tensor data32(ov::element::f32, shape);
        ov::Tensor expected_out(ov::element::f16, shape);

        auto in_ptr = static_cast<uint16_t*>(data.data());
        auto out_ptr = static_cast<uint16_t*>(expected_out.data());

        in_ptr[0]  = 0b0100010000000000;  // 4
        out_ptr[0] = 0b0100010000000000; // 4

        in_ptr[1]  = 0b1100010000000000;  // -4
        out_ptr[1] = 0b1100010000000000;  // -4

        in_ptr[2]  = 0b1100010000000001; // -4.0000...
        out_ptr[2] = 0b1100010000000000; // -4

        in_ptr[3]  = 0b1100010000000101; // -4.000...
        out_ptr[3] = 0b1100010000000000; // -4

        in_ptr[4]  = 0b1100010010000000;// -4.5 -> -4.0 (0b 1s10001e0010000000) - grs=100, last mantissa bit is 0, do nothing
        out_ptr[4] = 0b1100010000000000;

        in_ptr[5]  = 0b1100010110000000;  // -5.5 -> -6 1s10001e0110000000 - grs=100, last mantissa bit is 1, add 1 to lower mantissa bit
        out_ptr[5] = 0b1100011000000000;

        in_ptr[6]  = 0b1100010010100000;  // 1s10001e(01 101 00000)m - grs=101, last mantissa bit is 0, add 1 to lower mantissa bit
        out_ptr[6] = 0b1100010100000000;

        in_ptr[7]  = 0b1100010011000000;  // 1s10001e(00 110 000000)m - grs=110, last mantissa bit is 0, add 1 to lower mantissa bit
        out_ptr[7] = 0b1100010100000000;

        in_ptr[8]  = 0b1100010111100000;  // 1s10001e(01 111 00000) - grs=111, last mantissa bit is 1, add 1 to lower mantissa bit
        out_ptr[8] = 0b1100011000000000;

        ngraph::runtime::reference::convert(data.data<ov::float16>(),
                                            data32.data<float>(), sz);


        print_tensor(data);
        print_tensor(data32);

        std::map<std::string, ov::Any> config;
        auto compiled_model = core.compile_model(network, config);

        auto infer_request = compiled_model.create_infer_request();

        infer_request.set_input_tensor(0, data32);
        infer_request.infer();

        auto out_data = infer_request.get_output_tensor();

        ngraph::runtime::reference::convert(out_data.data<float>(), data.data<ov::float16>() , sz);

        print_tensor(data);
        print_tensor(out_data);

        cmp_mat(expected_out, data);
    }

    /// <summary>
    /// 1s-4e-3m - number of exponent bits less than in f16 and exponent bias is 11
    /// density of numbers near zero is highter than for bf16
    /// </summary>
    void test_hf8() {
        std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,<shape>,1,1" element_type="f32"/>
            <output>
                <port id="0" precision="FP32" names="in_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ConvertFP8" version="extension" destination_type="hf8">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP16" names="out_data">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim><shape></dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

        const int sz = 8;
        ov::Tensor weights;
        replace_all(model, "<shape>", std::to_string(sz));
        std::string deviceName = "CPU";
        auto network = core.read_model(model, weights);

        ov::Shape shape{1, sz, 1, 1};
        ov::Tensor data(ov::element::f16, shape);
        ov::Tensor data32(ov::element::f32, shape);
        ov::Tensor expected_out(ov::element::f16, shape);

        auto in_ptr = static_cast<uint16_t*>(data.data());
        auto out_ptr = static_cast<uint16_t*>(expected_out.data());

        in_ptr[0]  = 0b0100110000000000; // 16 must be clamp to 15
        out_ptr[0] = 0b0100101110000000;

        in_ptr[1]  = 0b0000000000111111; // denormalized value must be converted to zero
        out_ptr[1] = 0b0000000000000000;

        in_ptr[2]  = 0b0100101101000000;  // 14.525 0s10010e(110 100 0011) - exp = 3, grs = 100, mantisa bit is 0 -> do nothing
        out_ptr[2] = 0b0100101100000000;

        in_ptr[3] =  0b1100101101000000;  // -14.525 the same
        out_ptr[3] = 0b1100101100000000;

        in_ptr[4]  = 0b1100110000000000;  // -16 must be clamp to -15
        out_ptr[4] = 0b1100101110000000;

        in_ptr[5]  = 0b0101000000000000;  // 32 must be clamp to 15
        out_ptr[5] = 0b0100101110000000;

        in_ptr[6]  = 0b0100101101010011;  // 0s10010e(110 101 0011) - exp = 3, grs = 101, mantisa bit is 0 -> add one bit for mantissa
        out_ptr[6] = 0b0100101110000000;

        in_ptr[7]  = 0b0100101011010011;  // 0s10010e(101 101 0011) - exp = 3, grs = 101, mantisa bit is 1 -> add one bit for mantissa
        out_ptr[7] = 0b0100101100000000;

        ngraph::runtime::reference::convert(data.data<ov::float16>(), data32.data<float>(), sz);
        std::map<std::string, ov::Any> config;
        auto compiled_model = core.compile_model(network, config);
        auto infer_request = compiled_model.create_infer_request();

        infer_request.set_input_tensor(0, data32);
        infer_request.infer();

        auto out_data = infer_request.get_output_tensor();

        ngraph::runtime::reference::convert(out_data.data<float>(), data.data<ov::float16>(), sz);

        cmp_mat(expected_out, data);
    }
};

namespace {

std::string getOVExtensionPath() {
    return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
        std::string("openvino_template_extension") + IE_BUILD_POSTFIX);
}

}  // namespace

TEST_F(OVExtensionTests, LoadIR) {
    core.add_extension(getOVExtensionPath());
    test();
}

TEST_F(OVExtensionTests, LoadIR2) {
    core.add_extension(getOVExtensionPath());
    test_load_ir();
}

TEST_F(OVExtensionTests, InferBF8) {
    core.add_extension(getOVExtensionPath());
    test_bf8();
}

TEST_F(OVExtensionTests, InferHF8) {
    core.add_extension(getOVExtensionPath());
    test_hf8();
}
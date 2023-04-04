from typing import Mapping, OrderedDict
from transformers.onnx import OnnxConfig

model_id = "gpt_neox_model"

class GPTNeoXForCausalLMOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("past_key_values", {0: "32", 1: "1", 2: "batch", 3: "sequence_past", 4: "80"}), #[32,2,1,32,128,80]
            ]
        )

# class GPTNeoXForCausalLMOnnxConfig(OnnxConfig):
#     @property
#     def inputs(self) -> Mapping[str, Mapping[int, str]]:
#         return OrderedDict(
#             [
#                 ("input_ids", {0: 1, 1: 128}),
#                 ("past_key_values", {0: 32, 1: 1, 2: 1, 3: 128, 4: 80}), #[32,2,1,32,128,80]
#             ]
#         )


from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_id)
onnx_config = GPTNeoXForCausalLMOnnxConfig(config)

from pathlib import Path
from transformers.onnx import export
from transformers import AutoTokenizer, AutoModelForCausalLM

onnx_path = Path("gpt_neox_onnx/model.onnx")

base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_inputs, onnx_outputs = export(tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
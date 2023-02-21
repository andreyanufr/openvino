from openvino import runtime as ov
from openvino.tools.pot.graph import save_model
from openvino.runtime import Core, get_version, PartialShape

core = ov.Core()
target_shape = [1, 1024]
path_to_model = "/home/aanufriev/int8/transformer_experiments/notebooks/model/gpt_2_sq.xml"
path_to_model_reshaped = "/home/aanufriev/int8/transformer_experiments/notebooks/model/gpt_2_sq_reshape.xml"

# path_to_model = "/home/aanufriev/int8/transformer_experiments/notebooks/model/gpt_2_int8.xml"
# path_to_model_reshaped = "/home/aanufriev/int8/transformer_experiments/notebooks/model/gpt_2_int8_reshape.xml"

model = core.read_model(path_to_model)
model.reshape({"input_ids": PartialShape(target_shape)})

ov.serialize(model, path_to_model_reshaped)
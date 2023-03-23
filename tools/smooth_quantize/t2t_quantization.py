import os
import sys
#sys.path.append('/localdisk/xxiaofan/repo/fp8_exp/nncf')
from nncf.data.dataset import Dataset
from nncf.quantization.quantize import quantize
from pathlib import Path

import re
import numpy as np
import copy
import torch

import onnx
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import openvino.runtime as ov
from openvino.tools.pot import Metric, IEEngine, load_model, save_model, compress_model_weights, create_pipeline
from addict import Dict

DEFAULT_VAL_THREADS = 4
def get_torch_dataloader(folder, transform, batch_size=1):
    val_dataset = datasets.ImageFolder(root=folder, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )
    return val_loader, val_dataset

def create_timm_model(name):
    model = timm.create_model(
        name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path=""
    )
    return model

def get_model_transform(model):
    config = model.default_cfg
    config["mean"] = (0, 0, 0)
    config["std"] = (1.0, 1.0, 1.0)
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
    input_size = config["input_size"]
    resize_size = tuple(int(x / config["crop_pct"]) for x in input_size[-2:])

    RESIZE_MODE_MAP = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }

    transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=RESIZE_MODE_MAP["bicubic"]
            ),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            #transforms.invert(),
            normalize,
        ]
    )

    return transform

def benchmark_ov_model(model, dataloader, model_name, output_path):
    # Dump model
    ov_path = Path(output_path) / (model_name + ".xml")
    ov.serialize(model, str(ov_path))

    # Validate accuracy
    accuracy = validate_accuracy(ov_path, dataloader)
    return accuracy


def validate_accuracy(model_path, val_loader):
    dataset_size = len(val_loader)
    predictions = [0] * dataset_size
    references = [-1] * dataset_size

    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model)

    jobs = int(os.environ.get("NUM_VAL_THREADS", DEFAULT_VAL_THREADS))
    infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

    def process_result(request, userdata):
        output_data = request.get_output_tensor().data
        predicted_label = np.argmax(output_data, axis=1)
        predictions[userdata] = [predicted_label]

    infer_queue.set_callback(process_result)

    for i, (images, target) in tqdm(enumerate(val_loader)):
        # W/A for memory leaks when using torch DataLoader and OpenVINO
        permute = [2, 1, 0]
        images = images[:, permute]
        image_copies = copy.deepcopy(255 * images.numpy())
        infer_queue.start_async(image_copies, userdata=i)
        references[i] = target

    infer_queue.wait_all()
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)

    return accuracy_score(predictions, references)

def benchmark_torch_model(ov_path, dataloader):
    print(ov_path)
    #return 0.8112
    # Validate accuracy
    accuracy = validate_accuracy(ov_path, dataloader)
    return accuracy

#vit_base_patch32_224
torch.multiprocessing.set_sharing_strategy("file_system")  # W/A to avoid RuntimeError

data = "/home/aanuf/dataset/imagenet_loc/val_5" # "/home/aanufriev/dataset/imagenet_loc/val_5"
from nncf import IgnoredScope


model_name = "t2t_vit_14"
model_quantization_params = {"ignored_scope": IgnoredScope(names=[]), "subset_size": 350}

# model_quantization_params = {}
output = "./tmp_vit"
output = "./tmp_t2t"


output_folder = Path(output)
output_folder.mkdir(parents=True, exist_ok=True)
model = create_timm_model("vit_base_patch32_224")
model.eval().cpu()
transform = get_model_transform(model)

root_path = "/home/aanuf/int8__/transformer_experiments/notebooks/t2t"
ov_path = "/home/aanuf/int8__/transformer_experiments/notebooks/t2t/t2t-vit-14.xml"

batch_one_dataloader, dataset = get_torch_dataloader(data, transform, batch_size=1)
# batch_one_dataloader, dataset = get_torch_dataloader(data, transform, batch_size=1)
# # benchmark original models (once)
# orig_acc = benchmark_torch_model(
#     ov_path, batch_one_dataloader
# )

# print("FP32: ", orig_acc)


model_config = Dict({"model_name": "bert_mrpc", "model": ov_path, "weights": ov_path.replace('.xml', '.bin')})

stat_subset_size = 350

# layer_number = -1
# if len(sys.argv) > 1:
#     layer_number = int(sys.argv[1])
algorithms_sq = [
    {
        "name": "SmoothQuantize",
        "params": {
            "target_device": "ANY",
            "model_type": "transformer",
            "preset": "performance",
            "stat_subset_size": stat_subset_size,
            "alpha": 0.15,
            #"use_grid_alpha": True,
        },
    },
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "ANY",
            "model_type": "transformer",
            "preset": "performance",
            "stat_subset_size": stat_subset_size,
        },
    }
]

algorithms = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "ANY",
            "model_type": "transformer",
            "preset": "performance",
            "stat_subset_size": stat_subset_size,
        },
    }
]

model = load_model(model_config=model_config)
print(f"Load model {ov_path}")
engine_config = Dict({"device": "CPU"})
engine = IEEngine(config=engine_config, data_loader=dataset)
sq_pipeline = create_pipeline(algo_config=algorithms_sq, engine=engine)
sq_dir = f"{root_path}/{model_name}_INT8_SQ"
Path(sq_dir).mkdir(parents=True, exist_ok=True)
sq_compressed_model = sq_pipeline.run(model=model)
save_model(model=sq_compressed_model, save_path=sq_dir, model_name="openvino_model")


model = load_model(model_config=model_config)
print(f"Load model {model_name}")
engine = IEEngine(config=engine_config, data_loader=dataset)
q_pipeline = create_pipeline(algo_config=algorithms, engine=engine)
q_dir = f"{root_path}/{model_name}_INT8_POT"
Path(q_dir).mkdir(parents=True, exist_ok=True)
q_compressed_model = q_pipeline.run(model=model)
save_model(model=q_compressed_model, save_path=q_dir, model_name="openvino_model")


for precision in ["INT8_SQ", "INT8_POT"]:
    acc = benchmark_torch_model(f"{root_path}/{model_name}_{precision}/openvino_model.xml",  batch_one_dataloader)
    print(precision, acc)
#!/usr/bin/env python
# coding: utf-8

# # Quantize NLP models with Post-Training Optimization Tool ​in OpenVINO™
# This tutorial demonstrates how to apply `INT8` quantization to the Natural Language Processing model known as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), using the [Post-Training Optimization Tool API](https://docs.openvino.ai/latest/pot_compression_api_README.html) (part of the [OpenVINO Toolkit](https://docs.openvino.ai/)). A fine-tuned [HuggingFace BERT](https://huggingface.co/transformers/model_doc/bert.html) [PyTorch](https://pytorch.org/) model, trained on the [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398), will be used. The tutorial is designed to be extendable to custom models and datasets. It consists of the following steps:
# 
# - Download and prepare the BERT model and MRPC dataset.
# - Define data loading and accuracy validation functionality.
# - Prepare the model for quantization.
# - Run optimization pipeline.
# - Load and test quantized model.
# - Compare the performance of the original, converted and quantized models.

# ## Imports

# In[ ]:


import os
import sys
import time
import warnings
from pathlib import Path
from zipfile import ZipFile

import urllib
import urllib.parse
import urllib.request
from os import PathLike
from pathlib import Path
from tqdm.notebook import tqdm_notebook

import numpy as np
import torch
from addict import Dict
from compression.api import DataLoader as POTDataLoader
from compression.api import Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from openvino import runtime as ov
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

class DownloadProgressBar(tqdm_notebook):
    """
    TQDM Progress bar for downloading files with urllib.request.urlretrieve
    """

    def update_to(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if downloaded <= total_size:
            self.update(downloaded - self.n)


def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
    silent: bool = False,
    timeout: int = 10,
) -> str:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urlobject = urllib.request.urlopen(url, timeout=timeout)
        if filename is None:
            filename = urlobject.info().get_filename() or Path(urllib.parse.urlparse(url).path).name
    except urllib.error.HTTPError as e:
        raise Exception(f"File downloading failed with error: {e.code} {e.msg}") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, socket.timeout):
            raise Exception(
                "Connection timed out. If you access the internet through a proxy server, please "
                "make sure the proxy is set in the shell from where you launched Jupyter. If your "
                "internet connection is slow, you can call `download_file(url, timeout=30)` to "
                "wait for 30 seconds before raising this error."
            ) from None
        else:
            raise

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    # download the file if it does not exist, or if it exists with an incorrect file size
    urlobject_size = int(urlobject.info().get("Content-Length", 0))
    if not filename.exists() or (os.stat(filename).st_size != urlobject_size):
        progress_callback = DownloadProgressBar(
            total=urlobject_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        )
        urllib.request.urlretrieve(url, filename, reporthook=progress_callback.update_to)
        if os.stat(filename).st_size >= urlobject_size:
            progress_callback.update(urlobject_size - progress_callback.n)
            progress_callback.refresh()
    else:
        if not silent:
            print(f"'{filename}' already exists.")
    return filename.resolve()



# ## Settings

# In[ ]:


# Set the data and model directories, source URL and the filename of the model.
DATA_DIR = "data"
MODEL_DIR = "model"
MODEL_LINK = "https://download.pytorch.org/tutorial/MRPC.zip"
FILE_NAME = MODEL_LINK.split("/")[-1]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ## Prepare the Model
# Perform the following:
# - Download and unpack pre-trained BERT model for MRPC by PyTorch.
# - Convert the model to the ONNX.
# - Run Model Optimizer to convert the model from the ONNX representation to the OpenVINO Intermediate Representation (OpenVINO IR)

# In[ ]:


download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)
with ZipFile(f"{MODEL_DIR}/{FILE_NAME}", "r") as zip_ref:
    zip_ref.extractall(MODEL_DIR)


# Import all dependencies to load the original PyTorch model and convert it to the ONNX representation.

# In[ ]:


BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128


def export_model_to_onnx(model, path):
    with torch.no_grad():
        default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
        inputs = {
            "input_ids": default_input,
            "attention_mask": default_input,
            "token_type_ids": default_input,
        }
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            path,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids", "input_mask", "segment_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "input_mask": symbolic_names,
                "segment_ids": symbolic_names,
            },
        )
        print("ONNX model saved to {}".format(path))


torch_model = BertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "MRPC"))
onnx_model_path = Path(MODEL_DIR) / "bert_mrpc.onnx"
if not onnx_model_path.exists():
    export_model_to_onnx(torch_model, onnx_model_path)


# ## Convert the ONNX Model to OpenVINO IR

# In[ ]:


ir_model_xml = onnx_model_path.with_suffix(".xml")
ir_model_bin = onnx_model_path.with_suffix(".bin")

# Convert the ONNX model to OpenVINO IR FP32.
if not ir_model_xml.exists():
    os.system(f'mo --input_model {onnx_model_path} --output_dir {MODEL_DIR} --model_name {ir_model_xml.stem} --input input_ids,input_mask,segment_ids --input_shape [1,128],[1,128],[1,128] --output output --data_type FP32')


# ## Prepare MRPC Task Dataset
# 
# To run this tutorial, you will need to download the General Language Understanding Evaluation  (GLUE) data for the MRPC task from HuggingFace. Use the code below to download a script that fetches the MRPC dataset.

# In[ ]:


download_file(
    "https://raw.githubusercontent.com/huggingface/transformers/f98ef14d161d7bcdc9808b5ec399981481411cc1/utils/download_glue_data.py",
    show_progress=False,
)


# In[ ]:


from download_glue_data import format_mrpc

format_mrpc(DATA_DIR, "")


# ## Define DataLoader for POT
# In this step, you define `DataLoader` based on POT API. It will be used to collect statistics for quantization and run model evaluation. 
# Use helper functions from the HuggingFace Transformers to do the data preprocessing. It takes raw text data and encodes sentences and words, producing three model inputs. 
# For more details about the data preprocessing and tokenization, refer to this [description](https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca).

# In[ ]:


class MRPCDataLoader(POTDataLoader):
    # Required methods
    def __init__(self, config):
        """Constructor
        :param config: data loader specific config
        """
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self._task = config["task"].lower()
        self._model_dir = config["model_dir"]
        self._data_dir = config["data_source"]
        self._batch_size = config["batch_size"]
        self._max_length = config["max_length"]
        self.examples = []
        self._prepare_dataset()

    def __len__(self):
        """Returns size of the dataset"""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns annotation, data and metadata at the specified index.
        Possible formats:
        (index, annotation), data
        (index, annotation), data, metadata
        """
        if index >= len(self):
            raise IndexError

        batch = self.dataset[index]
        batch = tuple(t.detach().cpu().numpy() for t in batch)
        inputs = {"input_ids": batch[0], "input_mask": batch[1], "segment_ids": batch[2]}
        labels = batch[3]
        return (index, labels), inputs

    # Methods specific to the current implementation
    def _prepare_dataset(self):
        """Prepare dataset"""
        tokenizer = BertTokenizer.from_pretrained(self._model_dir, do_lower_case=True)
        processor = processors[self._task]()
        output_mode = output_modes[self._task]
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(self._data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=self._max_length,
            output_mode=output_mode,
        )
        all_input_ids = torch.unsqueeze(torch.tensor([f.input_ids for f in features], dtype=torch.long), 1)
        all_attention_mask = torch.unsqueeze(torch.tensor([f.attention_mask for f in features], dtype=torch.long), 1)
        all_token_type_ids = torch.unsqueeze(torch.tensor([f.token_type_ids for f in features], dtype=torch.long), 1)
        all_labels = torch.unsqueeze(torch.tensor([f.label for f in features], dtype=torch.long), 1)
        self.dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
        self.examples = examples


# ## Define Accuracy Metric Calculation
# In this step the `Metric` interface for MRPC task metrics is implemented. It is used for validating the accuracy of the models.

# In[ ]:


class Accuracy(Metric):

    # Required methods
    def __init__(self):
        super().__init__()
        self._name = "Accuracy"
        self._matches = []

    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        """Returns accuracy metric value for all model outputs."""
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """
        Updates prediction matches.

        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception(
                "The accuracy metric cannot be calculated " "for a model with multiple outputs"
            )
        output = np.argmax(output)
        match = output == target[0]
        self._matches.append(match)

    def reset(self):
        """
        Resets collected matches
        """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-better", "type": "accuracy"}}


# ## Run Quantization Pipeline
# Define a configuration for the quantization pipeline and run it. Keep in mind that built-in `IEEngine` implementation of `Engine` interface from the POT API for model inference is used here.

# In[ ]:


warnings.filterwarnings("ignore")  # Suppress accuracychecker warnings.

model_config = Dict({"model_name": "bert_mrpc", "model": ir_model_xml, "weights": ir_model_bin})
engine_config = Dict({"device": "CPU"})
dataset_config = {
    "task": "mrpc",
    "data_source": os.path.join(DATA_DIR, "MRPC"),
    "model_dir": os.path.join(MODEL_DIR, "MRPC"),
    "batch_size": BATCH_SIZE,
    "max_length": MAX_SEQ_LENGTH,
}
algorithms = [
    {
        "name": "MinMaxQuantization", #"DefaultQuantization"
        "params": {
            "target_device": "ANY",
            "model_type": "transformer",
            "preset": "performance",
            "stat_subset_size": 250,
        },
    }
]

core = ov.Core()
core.add_extension("/home/alex/work/openvino/openvino/bin/intel64/Release/libopenvino_template_extension.so")


# Step 1: Load the model.
model = load_model(model_config=model_config)

# Step 2: Initialize the data loader.
data_loader = MRPCDataLoader(config=dataset_config)

# Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
metric = Accuracy()

# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)
engine._ie.add_extension("/home/alex/work/openvino/openvino/bin/intel64/Release/libopenvino_template_extension.so")

# Step 5: Create a pipeline of compression algorithms.
pipeline = create_pipeline(algo_config=algorithms, engine=engine)

# Step 6 (Optional): Evaluate the original model. Print the results.
fp_results = pipeline.evaluate(model=model)
if fp_results:
    print("FP32 model results:")
    for name, value in fp_results.items():
        print(f"{name}: {value:.5f}")


# In[ ]:


# Step 7: Execute the pipeline.
warnings.filterwarnings("ignore")  # Suppress accuracychecker warnings.
print(
    f"Quantizing model with {algorithms[0]['params']['preset']} preset and {algorithms[0]['name']}"
)
start_time = time.perf_counter()
compressed_model = pipeline.run(model=model)
end_time = time.perf_counter()
print(f"Quantization finished in {end_time - start_time:.2f} seconds")

# Step 8 (Optional): Compress model weights to quantized precision
#                    in order to reduce the size of the final .bin file.
#compress_model_weights(model=compressed_model)

# Step 9: Save the compressed model to the desired path.
compressed_model_paths = save_model(model=compressed_model, save_path=MODEL_DIR, model_name="quantized_bert_mrpc"
)
compressed_model_xml = compressed_model_paths[0]["model"]


# In[ ]:


# Step 10 (Optional): Evaluate the compressed model and print the results.
int_results = pipeline.evaluate(model=compressed_model)

if int_results:
    print("INT8 model results:")
    for name, value in int_results.items():
        print(f"{name}: {value:.5f}")


# ## Load and Test OpenVINO Model
# 
# To load and test converted model, perform the following:
# * Load the model and compile it for CPU.
# * Prepare the input.
# * Run the inference.
# * Get the answer from the model output.

# In[ ]:

# Read the model from files.
model = core.read_model(model=compressed_model_xml)

# Assign dynamic shapes to every input layer.
for input_layer in model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[1] = -1
    model.reshape({input_layer: input_shape})

# Compile the model for a specific device.
compiled_model_int8 = core.compile_model(model=model, device_name="CPU")

output_layer = compiled_model_int8.outputs[0]


# The Data Loader returns a pair of sentences (indicated by `sample_idx`) and the inference compares these sentences and outputs whether their meaning is the same. You can test other sentences by changing `sample_idx` to another value (from 0 to 407).

# In[ ]:


sample_idx = 5

sample = data_loader.examples[sample_idx]
inputs = data_loader[sample_idx][1]

result = compiled_model_int8(inputs)[output_layer]
result = np.argmax(result)

print(f"Text 1: {sample.text_a}")
print(f"Text 2: {sample.text_b}")
print(f"The same meaning: {'yes' if result == 1 else 'no'}")


# ## Compare Performance of the Original, Converted and Quantized Models
# 
# Compare the original PyTorch model with OpenVINO converted and quantized models (`FP32`, `INT8`) to see the difference in performance. It is expressed in Sentences Per Second (SPS) measure, which is the same as Frames Per Second (FPS) for images.

# In[ ]:


model = core.read_model(model=ir_model_xml)

# Assign dynamic shapes to every input layer.
for input_layer in model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[1] = -1
    model.reshape({input_layer: input_shape})

# Compile the model for a specific device.
compiled_model_fp32 = core.compile_model(model=model, device_name="CPU")


# In[ ]:


num_samples = 50
inputs = data_loader[0][1]

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(num_samples):
        torch_model(torch.as_tensor(list(inputs.values())).squeeze())
    end = time.perf_counter()
    time_torch = end - start
print(
    f"PyTorch model on CPU: {time_torch / num_samples:.3f} seconds per sentence, "
    f"SPS: {num_samples / time_torch:.2f}"
)

start = time.perf_counter()
for _ in range(num_samples):
    compiled_model_fp32(inputs)
end = time.perf_counter()
time_ir = end - start
print(
    f"IR FP32 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} "
    f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
)

start = time.perf_counter()
for _ in range(num_samples):
    compiled_model_int8(inputs)
end = time.perf_counter()
time_ir = end - start
print(
    f"OpenVINO IR INT8 model in OpenVINO Runtime/CPU: {time_ir / num_samples:.3f} "
    f"seconds per sentence, SPS: {num_samples / time_ir:.2f}"
)


# Finally, measure the inference performance of OpenVINO `FP32` and `INT8` models. For this purpose, use [Benchmark Tool](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html) in OpenVINO.
# 
# > **Note**: The `benchmark_app` tool is able to measure the performance of the OpenVINO Intermediate Representation (OpenVINO IR) models only. For more accurate performance, run `benchmark_app` in a terminal/command prompt after closing other applications. Run `benchmark_app -m model.xml -d CPU` to benchmark async inference on CPU for one minute. Change `CPU` to `GPU` to benchmark on GPU. Run `benchmark_app --help` to see an overview of all command-line options.

# In[ ]:


# Inference FP32 model (OpenVINO IR)
#os.system(' benchmark_app -m $ir_model_xml -d CPU -api sync')


# In[ ]:


# Inference INT8 model (OpenVINO IR)
#os.system(' benchmark_app -m $compressed_model_xml -d CPU -api sync')


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
from datasets import load_dataset

from openvino.tools.pot import DataLoader as POTDataLoader
from openvino.tools.pot import Metric, IEEngine, load_model, save_model
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.runtime import PartialShape
# from compression.pipeline.initializer import create_pipeline
from openvino import runtime as ov
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# Set the data and model directories, source URL and the filename of the model.
DATA_DIR = "data"
MODEL_DIR = "gpt_neox_ov_model_spr"
MODEL_LINK = "https://download.pytorch.org/tutorial/MRPC.zip"
FILE_NAME = MODEL_LINK.split("/")[-1]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128

calibration_dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train[:1000]')
# calibration_dataset = load_dataset('squad', split='train[:1000]')


# ## Convert the ONNX Model to OpenVINO IR

# In[ ]:


ir_model_xml = "/home/aanuf/gpt_neox/gpt_neox_ov_model/gpt_neox_model_.xml"
ir_model_bin = "/home/aanuf/gpt_neox/gpt_neox_ov_model/gpt_neox_model_.bin"


class MRPCDataLoader(POTDataLoader):
    # Required methods
    def __init__(self, dataset, tokenizer):
        """Constructor
        :param config: data loader specific config
        """
        self.dataset = dataset
        self.tokenizer = tokenizer

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
        inputs = self.tokenizer(batch)

        return (index, 0), inputs


class TextPreprocessor:
    def __init__(self, tokenizer, max_len=MAX_SEQ_LENGTH):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self.mean_len = 0
        self.n_calls = 0

    def __call__(self, data_item):
        input_ids = self._tokenizer(data_item['text'])
        # input_ids = self._tokenizer(data_item['context'])

        input_ids = np.array(input_ids['input_ids'])

        self.n_calls += 1
        self.mean_len = ((self.n_calls - 1) * self.mean_len + input_ids.shape[0]) / self.n_calls

        if input_ids.shape[0] >= self._max_len:
            input_ids = input_ids[:self._max_len]

        pad_len = self._max_len - input_ids.shape[0]
        input_ids = np.pad(input_ids, pad_width=(0, pad_len), mode='constant', constant_values=1)
        input_ids = np.expand_dims(input_ids, axis=0)

        attention_mask = np.ones((1, self._max_len), dtype=float)

        attention_mask[:] = 1
        if pad_len > 0:
            attention_mask[:, -pad_len:] = 0.0

        past_key_values = np.zeros((32, 2, 1, 32, 1, 80), dtype=float)

        return {'input_ids': input_ids, "past_key_values.1": past_key_values}


def build_tokenizer(model_id, model_max_length=MAX_SEQ_LENGTH):
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=model_max_length)

    return tokenizer


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

model_config = Dict({"model_name": "gpt_neox_model", "model": ir_model_xml, "weights": ir_model_bin})
engine_config = Dict({"device": "CPU"})

# alpha = 0.95

# if len(sys.argv) > 1:
#     alpha = float(sys.argv[1])

algorithms = [
    # {
    #     "name": "SmoothQuantize",
    #     "params": {
    #         "target_device": "ANY",
    #         "model_type": "transformer",
    #         "preset": "performance",
    #         "stat_subset_size": 350,
    #         "alpha": alpha,
    #         "saturation_fix": "no",
    #     },
    # },
    {
        "name": "DefaultQuantization",  # "DefaultQuantization",
        "params": {
            "target_device": "CPU_SPR",
            "model_type": "transformer",
            "preset": "performance",
            "stat_subset_size": 350,
            "saturation_fix": "no",
        },
    }
]

core = ov.Core()

max_len = MAX_SEQ_LENGTH
# Step 1: Load the model.
model = load_model(model_config=model_config)
tokenizer = build_tokenizer("gpt2", model_max_length=max_len)

data_loader = MRPCDataLoader(calibration_dataset, TextPreprocessor(tokenizer, max_len=max_len))

tmp = data_loader.__getitem__(10)

# Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
metric = Accuracy()

# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

# Step 5: Create a pipeline of compression algorithms.
pipeline = create_pipeline(algo_config=algorithms, engine=engine)

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
# compress_model_weights(model=compressed_model)

# Step 9: Save the compressed model to the desired path.
compressed_model_paths = save_model(model=compressed_model, save_path=MODEL_DIR, model_name="gpt_neox_int8")
compressed_model_xml = compressed_model_paths[0]["model"]

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
from openvino.tools.pot.pipeline.initializer import create_pipeline
#from compression.pipeline.initializer import create_pipeline
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



warnings.filterwarnings("ignore")  # Suppress accuracychecker warnings.

engine_config = Dict({"device": "CPU"})
dataset_config = {
    "task": "mrpc",
    "data_source": os.path.join(DATA_DIR, "MRPC"),
    "model_dir": os.path.join(MODEL_DIR, "MRPC"),
    "batch_size": BATCH_SIZE,
    "max_length": MAX_SEQ_LENGTH,
}

core = ov.Core()


# Step 2: Initialize the data loader.
data_loader = MRPCDataLoader(config=dataset_config)

model_xml = "/home/aanufriev/int8/transformer_experiments/model/bert_mrpc.xml"
dump_dir = "bert_activations"

model_xml = "/home/aanufriev/int8/transformer_experiments/notebooks/model/int8_quantized_bert_mrpc.xml"
dump_dir = "bert_activations_q"

# Read the model from files.
model = core.read_model(model=model_xml)

ops_fp32 = model.get_ordered_ops()
names = []
names_map = {}
for op in ops_fp32:
    l_name = op.friendly_name
    names.append(l_name)
    names_map[op.name] = l_name
    if 'sq_mul' in l_name:
        print(l_name)
    try:
        model.add_outputs([op.output(0)])#[l_name])
    except:
        print("Skip node: {}".format(l_name))
        continue


compiled_model = core.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.outputs[0]


# The Data Loader returns a pair of sentences (indicated by `sample_idx`) and the inference compares these sentences and outputs whether their meaning is the same. You can test other sentences by changing `sample_idx` to another value (from 0 to 407).

# In[ ]:


dump_res = {}
n = 1

sample_idx = 5

sample = data_loader.examples[sample_idx]
inputs = data_loader[sample_idx][1]

result = compiled_model(inputs)#[output_layer]

results_v = iter(result.values())
results_k = iter(result.keys())

for k32 in results_k:
    v32 = next(results_v)
    try:
        tmp = k32.any_name
    except:
        k = k32.node.friendly_name
        print(k)
        if not k in dump_res:
            dump_res[k] = v32
        else:
            dump_res[k] = ((n - 1) * dump_res[k] + v32) / n
        continue
        # pass
    
    if not k32.any_name in dump_res:
        dump_res[k32.any_name] = v32
    else:
        dump_res[k32.any_name] = ((n - 1) * dump_res[k32.any_name] + v32) / n


if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

for k, v in dump_res.items():
    np.save(os.path.join(dump_dir, k.replace('/', '_')+".npy"), v)


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
from torch.utils.data import TensorDataset

import numpy as np
import torch
from addict import Dict

from openvino import runtime as ov
from openvino.tools.pot import DataLoader as POTDataLoader
from openvino.tools.pot import Metric, IEEngine, load_model, save_model, compress_model_weights, create_pipeline


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from datasets import load_dataset


BATCH_SIZE = 1
MAX_SEQ_LENGTH = 128

onnx_dir = "onnx"
openvino_dir = "openvino"

if 1: #TODO: remove
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

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)
    with ZipFile(f"{MODEL_DIR}/{FILE_NAME}", "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    
    from download_glue_data import format_mrpc

    format_mrpc(DATA_DIR, "")

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
            self._model_id = config["model_id"]
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
            tokenizer = AutoTokenizer.from_pretrained(self._model_id, do_lower_case=True) #BertTokenizer.from_pretrained(self._model_dir, do_lower_case=True)
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
            try:
                all_token_type_ids = torch.unsqueeze(torch.tensor([f.token_type_ids for f in features], dtype=torch.long), 1)
            except:
                all_token_type_ids = torch.zeros_like(all_attention_mask)

            all_labels = torch.unsqueeze(torch.tensor([f.label for f in features], dtype=torch.long), 1)
            self.dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_labels
            )
            self.examples = examples

def export_model_to_onnx(model, path):
    with torch.no_grad():
        default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
        default_input = torch.zeros(1, MAX_SEQ_LENGTH, dtype=torch.int64)

        inputs = {
            "input_ids": default_input,
            "attention_mask": default_input,
            "token_type_ids": default_input,
        }

        # inputs = {
        #     "input_ids": default_input,
        #     "attention_mask": default_input,
        # }

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


class MRPCDataLoader_(POTDataLoader):
    # Required methods
    def __init__(self, dataset, text_preprocessor):
        """Constructor
        :param config: data loader specific config
        """
        self.dataset = dataset
        self.text_preprocessor = text_preprocessor

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
        inputs = self.text_preprocessor(batch)

        return (index, batch['label']), inputs




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


calibration_dataset = load_dataset("glue", "mrpc", split='train[:1000]')
validation_dataset = load_dataset("glue", "mrpc", split='test')

def build_tokenizer(model_id, model_max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=model_max_length)

    return tokenizer

def build_tokenizer_and_model(model_id, model_max_length=512):
    tokenizer = build_tokenizer(model_id, model_max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    return tokenizer, model


def transform_fn(data_item):
    images, _ = data_item
    return images

class TextPreprocessor:
    def __init__(self, tokenizer, max_len=512):
        self._tokenizer = tokenizer
        self._max_len = max_len

    def __call__(self, data_item):
        data = self._tokenizer([(data_item['sentence1'], data_item['sentence2'])],
                                max_length=self._max_len,
                                padding="max_length",
                                truncation=True)

        # batch_encoding = tokenizer(
        #     [(example.text_a, example.text_b) for example in examples],
        #     max_length=max_length,
        #     padding="max_length",
        #     truncation=True,
        # )
        
        # input_ids = np.array(input_ids['input_ids'])
        # if input_ids.shape[0] >= self._max_len:
        #     input_ids = input_ids[:self._max_len]

        # pad_len = self._max_len - input_ids.shape[0]
        # input_ids = np.pad(input_ids, pad_width=(0, pad_len), mode='constant', constant_values=1)
        # input_ids = np.expand_dims(input_ids, axis=0)

        # attention_mask = np.ones((1, self._max_len), dtype=float)

        # attention_mask[:] = 1
        # attention_mask[:, -pad_len:] = 0.0

        if 'token_type_ids' in data.keys():
            return {'input_ids': np.array(data['input_ids']),
                    'input_mask': np.array(data['attention_mask']),
                    "segment_ids": np.array(data['token_type_ids'])}
        else:
            return {'input_ids': np.array(data['input_ids']),
                    'input_mask': np.array(data['attention_mask'])}


def build_preprocessing(model_id, model_max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=model_max_length)
    text_preprocessor = TextPreprocessor(tokenizer, model_max_length)

    return text_preprocessor


model_names = ["yoshitomo-matsubara/bert-large-uncased-mrpc",
               "bert-base-cased-finetuned-mrpc", 
               "textattack/roberta-base-MRPC",
               "textattack/bert-base-uncased-MRPC",
               "Intel/bert-base-uncased-mrpc",
               "M-FAC/bert-mini-finetuned-mrpc",
               "yoshitomo-matsubara/bert-base-uncased-mrpc",
               "textattack/distilbert-base-uncased-MRPC",
               "gchhablani/bert-base-cased-finetuned-mrpc",
               "gokuls/bert-base-uncased-mrpc",
               "Intel/xlnet-base-cased-mrpc",
               "Intel/bart-large-mrpc",
               "textattack/distilbert-base-cased-MRPC",
               "Intel/camembert-base-mrpc",
               "Intel/distilbert-base-cased-distilled-squad-int8-static",
               "anirudh21/albert-xlarge-v2-finetuned-mrpc",
               "Intel/MiniLM-L12-H384-uncased-mrpc-int8-static",
               "Intel/electra-small-discriminator-mrpc-int8-static",
               "Intel/MiniLM-L12-H384-uncased-mrpc",
               "Intel/electra-small-discriminator-mrpc",
               "textattack/xlnet-base-cased-MRPC"]

model_names = list(set(model_names))


ov_model_names = []

if not os.path.exists(onnx_dir):
    os.mkdir(onnx_dir)

if not os.path.exists(openvino_dir):
    os.mkdir(openvino_dir)

onnx_dir = os.path.join(os.getcwd(), onnx_dir)
openvino_dir = os.path.join(os.getcwd(), openvino_dir)

for model_name in model_names:
    model_id = model_name
    model_name = model_name.replace('/', '_')
    ov_model_name = os.path.join(openvino_dir, f"{model_name}.xml")

    if os.path.exists(ov_model_name):
        print(f"Find model from cashe: {model_id}")
        ov_model_names.append((model_id, ov_model_name))
        continue
    continue
    try:
        tokenizer, model = build_tokenizer_and_model(model_id, model_max_length=MAX_SEQ_LENGTH)
    except:
        print(f'Error in model or tokenizer loading for {model_id}')
    
    onnx_model_path = os.path.join(onnx_dir, model_name+".onnx")

    try:
        export_model_to_onnx(model, onnx_model_path)
        #os.system(f'python -m transformers.onnx --model={model_id} {onnx_dir}')
        os.system(f'mv {onnx_dir}/model.onnx {onnx_model_path}')
    except:
        print(f"Error in onnx convertion for model {model_name}")
        continue
    
    try:
        os.system(f'mo --input_model {onnx_model_path} --output_dir {openvino_dir} --model_name {model_name} --input input_ids,input_mask,segment_ids --input_shape [1,128],[1,128],[1,128] --output output --data_type FP32')
    except:
        print(f"Error in ir convertion for model {onnx_model_path}")
        continue

    ov_model_names.append((model_id, ov_model_name))

sys.stdout.flush()

warnings.filterwarnings("ignore")  # Suppress accuracychecker warnings.

core = ov.Core()

for model_id, model_name in ov_model_names:
    model_config = Dict({"model_name": "bert_mrpc", "model": model_name, "weights": model_name.replace('.xml', '.bin')})
    engine_config = Dict({"device": "CPU"})

    stat_subset_size = 350
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

    algorithms_sq = [
        {
            "name": "SmoothQuantize",
            "params": {
                "target_device": "ANY",
                "model_type": "transformer",
                "preset": "performance",
                "stat_subset_size": stat_subset_size,
                "alpha": 0.95
            },
        },
        {
            "name": "MinMaxQuantization",
            "params": {
                "target_device": "ANY",
                "model_type": "transformer",
                "preset": "performance",
                "stat_subset_size": stat_subset_size,
            },
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config=model_config)
    print(f"Load model {model_name}")
    sys.stdout.flush()

    # saved_model_paths = save_model(model=model, save_path=MODEL_DIR, model_name="no_quantized_bert_mrpc"
    # )

    # Step 2: Initialize the data loader.
    # text_preprocessor = build_preprocessing(model_id, model_max_length=MAX_SEQ_LENGTH)
    # data_loader = MRPCDataLoader(validation_dataset, text_preprocessor)

    dataset_config = {
        "task": "mrpc",
        "data_source": os.path.join(DATA_DIR, "MRPC"),
        "model_dir": os.path.join(MODEL_DIR, "MRPC"),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_SEQ_LENGTH,
        "model_id": model_id
    }
    data_loader = MRPCDataLoader(config=dataset_config)

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = Accuracy()

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    q_pipeline = create_pipeline(algo_config=algorithms, engine=engine)

    print(model_id)
    fp_results = q_pipeline.evaluate(model=model)
    if fp_results:
        print("FP32 model results:")
        for name, value in fp_results.items():
            print(f"{name}: {value:.5f}")

    start_time = time.perf_counter()
    compressed_model = q_pipeline.run(model=model)
    end_time = time.perf_counter()
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    model_id = model_id.replace('/', '_')
    compressed_model_paths = save_model(model=compressed_model, save_path=openvino_dir, model_name=model_id+"_int8")
    compressed_model_xml = compressed_model_paths[0]["model"]


    # Step 6 (Optional): Evaluate the compressed model and print the results.
    int_results = q_pipeline.evaluate(model=compressed_model)
    if int_results:
        print("INT8 model results:")
        for name, value in int_results.items():
            print(f"{name}: {value:.5f}")
    

    # Step 7
    model = load_model(model_config=model_config)
    print(f"Load model {model_name}")
    sys.stdout.flush()

    sq_pipeline = create_pipeline(algo_config=algorithms_sq, engine=engine)


    start_time = time.perf_counter()
    sq_compressed_model = sq_pipeline.run(model=model)
    end_time = time.perf_counter()
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    compressed_model_paths = save_model(model=sq_compressed_model, save_path=openvino_dir, model_name=model_id+"_int8_sq")
    compressed_model_xml = compressed_model_paths[0]["model"]


    # Step 6 (Optional): Evaluate the compressed model and print the results.
    sq_int_results = sq_pipeline.evaluate(model=sq_compressed_model)
    if sq_int_results:
        print("SQ INT8 model results:")
        for name, value in sq_int_results.items():
            print(f"{name}: {value:.5f}")
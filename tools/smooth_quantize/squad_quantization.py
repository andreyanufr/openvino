import time
from functools import partial
from pathlib import Path
import shutil
import os
import sys

from datasets import load_dataset, load_metric
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from optimum.intel.openvino import OVConfig, OVModelForQuestionAnswering
from optimum.intel.openvino.quantization import OVQuantizer

from openvino import runtime as ov
from openvino.tools.pot import DataLoader as POTDataLoader
from openvino.tools.pot import Metric, IEEngine, load_model, save_model, compress_model_weights, create_pipeline
from addict import Dict
import numpy as np
import torch

import random
random.seed(10)


class SquadDataLoader(POTDataLoader):
    # Required methods
    def __init__(self, dataset):
        """Constructor
        :param config: data loader specific config
        """
        self.dataset = dataset

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
        if "token_type_ids" in batch:
            inputs = {"input_ids": np.array(batch["input_ids"]),
                    "attention_mask": np.array(batch["attention_mask"]),
                    "token_type_ids": np.array(batch["token_type_ids"])}
        else:
            inputs = {"input_ids": np.array(batch["input_ids"]),
                      "attention_mask": np.array(batch["attention_mask"])}

        return (index, 0), inputs


def export_model_to_onnx(model, path, has_token_type_ids=True):
    MAX_SEQ_LENGTH = 512
    with torch.no_grad():
        default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)
        default_input = torch.zeros(1, MAX_SEQ_LENGTH, dtype=torch.int64)

            # inputs = {"input_ids": np.array(batch["input_ids"]),
            #         "attention_mask": np.array(batch["attention_mask"]),
            #         "token_type_ids": np.array(batch["token_type_ids"])}
        
        symbolic_names = {0: "batch_size", 1: "max_seq_len"}
        if has_token_type_ids:
            inputs = {
                "input_ids": default_input,
                "attention_mask": default_input,
                "token_type_ids": default_input,
            }
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
                path,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["start_logits", "end_logits"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                    "token_type_ids": symbolic_names,
                },
            )
        else:
            inputs = {
                "input_ids": default_input,
                "attention_mask": default_input,
            }
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                path,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["start_logits", "end_logits"],
                dynamic_axes={
                    "input_ids": symbolic_names,
                    "attention_mask": symbolic_names,
                },
            )
        

        print("ONNX model saved to {}".format(path))

def compute_metric_squad(model_id, precision):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForQuestionAnswering.from_pretrained(f"models/{model_id}_{precision}")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    dataset = load_dataset("squad", split="validation[:100]")
    metric = load_metric("squad")
    predictions = []
    references = []
    for item in dataset:
        prediction = qa_pipeline({"context": item["context"], "question": item["question"]})
        metric_prediction = {"id": item["id"], "prediction_text": prediction["answer"]}
        metric_reference = {
            "id": item["id"],
            "answers": {"answer_start": item["answers"]["answer_start"], "text": item["answers"]["text"]},
        }
        predictions.append(metric_prediction)
        references.append(metric_reference)
    return metric.compute(predictions=predictions, references=references)


def preprocess_fn_qa(examples, tokenizer):
    return tokenizer(
        examples["question"],
        examples["context"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )


def quantize(model_id, dataset_name, preprocess_function, dataset_config=None):
    # for precision in ["FP32", "INT8", "INT8_SQ", "INT8_POT"]:
    #     metrics = compute_metric_squad(model_id, precision)
    #     for metric_name, metric_value in metrics.items():
    #         print(f"{model_id},{precision},{metric_name}: {metric_value:.2f}")
    # return

    print(f"Quantizing {model_id}")
    openvino_dir = f"models/{model_id}_FP32"

    if not Path(openvino_dir).exists():
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        ov_model.save_pretrained(openvino_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        dataset_name,
        preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
        dataset_config_name=dataset_config,
        num_samples=350,
        dataset_split="train",
        preprocess_batch=True,
    )

    onnx_model_path = openvino_dir + "/model.onnx"
    model_name = model_id.replace('/', '_')
    item = calibration_dataset[0]
    has_token_type_ids = "token_type_ids" in item

    try:
        export_model_to_onnx(model, onnx_model_path, has_token_type_ids)
        #os.system(f'python -m transformers.onnx --model={model_id} {onnx_dir}')
        #os.system(f'mv {onnx_dir}/model.onnx {onnx_model_path}')
    except:
        print(f"Error in onnx convertion for model {model_name}")
        return
    
    try:
        if has_token_type_ids:
            os.system(f'mo --input_model {onnx_model_path} --output_dir {openvino_dir} --model_name {model_name} --input input_ids,attention_mask,token_type_ids --input_shape [1,512],[1,512],[1,512] --output start_logits,end_logits --data_type FP32')
        else:
            os.system(f'mo --input_model {onnx_model_path} --output_dir {openvino_dir} --model_name {model_name} --input input_ids,attention_mask --input_shape [1,512],[1,512] --output start_logits,end_logits --data_type FP32')
    except:
        print(f"Error in ir convertion for model {onnx_model_path}")
        return

    start_time = time.perf_counter()
    
    ov_config = OVConfig()
    # quantizer.quantize(
    #     save_directory=f"models/{model_id}_INT8",
    #     quantization_config=ov_config,
    #     calibration_dataset=calibration_dataset,
    #     batch_size=2,
    # )
    end_time = time.perf_counter()
    duration = end_time - start_time

    #model_name = f"models/{model_id}_FP32/openvino_model.xml"
    model_name = f"{openvino_dir}/{model_name}.xml"

    model_config = Dict({"model_name": "bert_mrpc", "model": model_name, "weights": model_name.replace('.xml', '.bin')})

    # smooth quantize + int8 pot
    model = load_model(model_config=model_config)
    print(f"Load model {model_name}")

    stat_subset_size = 350

    layer_number = -1
    if len(sys.argv) > 1:
        layer_number = int(sys.argv[1])
    algorithms_sq = [
        {
            "name": "SmoothQuantize",
            "params": {
                "target_device": "ANY",
                "model_type": "transformer",
                "preset": "performance",
                "stat_subset_size": stat_subset_size,
                "alpha": 0.95,
                "layer_number": layer_number,
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

    engine_config = Dict({"device": "CPU"})
    engine = IEEngine(config=engine_config, data_loader=SquadDataLoader(calibration_dataset))

    sq_pipeline = create_pipeline(algo_config=algorithms_sq, engine=engine)

    sq_dir = f"models/{model_id}_INT8_SQ"
    Path(sq_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    sq_compressed_model = sq_pipeline.run(model=model)
    end_time = time.perf_counter()
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    save_model(model=sq_compressed_model, save_path=sq_dir, model_name="openvino_model")
    
    shutil.copy(f"models/{model_id}_FP32/config.json", sq_dir)

    # int8 pot
    engine = IEEngine(config=engine_config, data_loader=SquadDataLoader(calibration_dataset))
    model = load_model(model_config=model_config)
    print(f"Load model {model_name}")

    q_pipeline = create_pipeline(algo_config=algorithms, engine=engine)

    q_dir = f"models/{model_id}_INT8_POT"
    Path(q_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    q_compressed_model = q_pipeline.run(model=model)
    end_time = time.perf_counter()
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    save_model(model=q_compressed_model, save_path=q_dir, model_name="openvino_model")
    
    shutil.copy(f"models/{model_id}_FP32/config.json", q_dir)

    with open("result_squad.csv", 'a') as f:
        #f.write("Model,FP32,INT8_SQ, INT8_POT")
        line = f"{model_id}"
        for precision in ["FP32", "INT8_SQ", "INT8_POT"]:#["FP32", "INT8", "INT8_SQ", "INT8_POT"]:
            metrics = compute_metric_squad(model_id, precision)
            for metric_name, metric_value in metrics.items():
                print(f"{model_id},{precision},{metric_name}: {metric_value:.2f}")
                line += f",{metric_value:.2f}"
        f.write(line + "\n")


def quantize_sq(model_id, dataset_name, preprocess_function, dataset_config=None):
    print(f"Quantizing {model_id}")
    openvino_dir = f"models/{model_id}_FP32"
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        dataset_name,
        preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
        dataset_config_name=dataset_config,
        num_samples=350,
        dataset_split="train",
        preprocess_batch=True,
    )

    onnx_model_path = openvino_dir + "/model.onnx"
    model_name = model_id.replace('/', '_')
    item = calibration_dataset[0]
    has_token_type_ids = "token_type_ids" in item

    #model_name = f"models/{model_id}_FP32/openvino_model.xml"
    model_name = f"{openvino_dir}/{model_name}.xml"

    model_config = Dict({"model_name": "bert_mrpc", "model": model_name, "weights": model_name.replace('.xml', '.bin')})
    model = load_model(model_config=model_config)
    print(f"Load model {model_name}")

    stat_subset_size = 350
    layer_number = -1
    if len(sys.argv) > 1:
        layer_number = int(sys.argv[1])
    algorithms_sq = [
        # {
        #     "name": "SmoothQuantize",
        #     "params": {
        #         "target_device": "ANY",
        #         "model_type": "transformer",
        #         "preset": "performance",
        #         "stat_subset_size": stat_subset_size,
        #         "alpha": 0.95,
        #         "layer_number": layer_number
        #     },
        # },
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

    engine_config = Dict({"device": "CPU"})
    engine = IEEngine(config=engine_config, data_loader=SquadDataLoader(calibration_dataset))

    sq_pipeline = create_pipeline(algo_config=algorithms_sq, engine=engine)

    sq_dir = f"models_new/{model_id}_INT8_SQ"
    Path(sq_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    sq_compressed_model = sq_pipeline.run(model=model)
    end_time = time.perf_counter()
    print(f"Quantization finished in {end_time - start_time:.2f} seconds")

    save_model(model=sq_compressed_model, save_path=sq_dir, model_name="openvino_model")

    # for precision in ["FP32", "INT8", "INT8_SQ", "INT8_POT"]:
    for precision in ["INT8_SQ"]:
        metrics = compute_metric_squad(model_id, precision)
        for metric_name, metric_value in metrics.items():
            print(f"{model_id},{precision},{metric_name}: {metric_value:.2f}")

# tried models with error
#"valhalla/longformer-base-4096-finetuned-squadv1", - longformer unsupported in optimum
#"mrm8488/spanbert-large-finetuned-squadv1", - error with load tokenizer
#"ramsrigouthamg/t5_squad_v1", #ValueError: Unrecognized configuration class <class 'transformers.models.t5.configuration_t5.T5Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
#"neuralmagic/oBERT-teacher-squadv1", # raise ValueError("The feature could not be extracted and needs to be specified for the ONNX export.")
#"sshleifer/tiny-distilbert-base-cased-distilled-squad", - accuracy is small for fp32
#"deepset/xlm-roberta-large-squad2",

with open("result_squad.csv", 'a') as f:
    f.write("Model,FP32,INT8_SQ,INT8_POT\n")

model_names = ["deepset/tinyroberta-squad2",
               "deepset/minilm-uncased-squad2", # - long max_len
               "dmis-lab/biobert-large-cased-v1.1-squad", #work
               "ktrapeznikov/biobert_v1.1_pubmed_squad_v2", # work
               "csarron/bert-base-uncased-squad-v1",
               "distilbert-base-uncased-distilled-squad",
               "twmkn9/distilroberta-base-squad2"]


for model in model_names:
    #quantize_sq(model, dataset_name="squad", preprocess_function=preprocess_fn_qa)
    quantize(model, dataset_name="squad", preprocess_function=preprocess_fn_qa)
    # try:
    #     quantize(model, dataset_name="squad", preprocess_function=preprocess_fn_qa)
    # except:
    #     print(f"Error in model {model}")
    #     pass

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from cv2 import imread, resize as cv2_resize
import openvino.runtime as ov

import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline
from openvino.tools.pot.utils.logger import init_logger
from openvino.tools.pot.api.samples.utils.argument_parser import get_common_argparser


import matplotlib.pyplot as plt
import timm


def create_timm_model(name):
    model = timm.create_model(
        name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path=""
    )
    return model

def get_model_transform(model):
    config = model.default_cfg
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
                resize_size, interpolation=RESIZE_MODE_MAP[config["interpolation"]]
            ),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform

# Initialize the logger to print the quantization process in the console.
init_logger(level='INFO')


ie = ov.Core()


# Custom DataLoader class implementation that is required for
# the proper reading of Imagenet images and annotations.
class ImageNetDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)
        self._annotations, self._img_ids = self._read_img_ids_annotations(self.config)

    def __len__(self):
        return len(self._img_ids)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        annotation = self._annotations[self._img_ids[index]] if self._annotations else None
        return self._read_image(self._img_ids[index]), annotation

    # Methods specific to the current implementation
    @staticmethod
    def _read_img_ids_annotations(dataset):
        """ Parses annotation file or directory with images to collect image names and annotations.
        :param dataset: dataset config
        :returns dictionary with annotations
                 list of image ids
        """
        annotations = {}
        img_ids = []
        if dataset.annotation_file:
            with open(dataset.annotation_file) as f:
                for line in f:
                    img_id, annotation = line.split(" ")
                    annotation = int(annotation.rstrip('\n'))
                    annotations[img_id] = annotation + 1 if dataset.has_background else annotation
                    img_ids.append(img_id)
                    if len(img_ids) > 1000:
                        break
        else:
            img_ids = sorted(os.listdir(dataset.data_source))

        return annotations, img_ids

    def _read_image(self, index):
        """ Reads images from directory.
        :param index: image index to read
        :return ndarray representation of image batch
        """
        image = imread(os.path.join(self.config.data_source, index))
        #print(np.max(image))
        image = self._preprocess(image)/ 255.0
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        image = (image - mean) / std
        return image

    def _preprocess(self, image):
        """ Does preprocessing of an image according to the preprocessing config.
        :param image: ndarray image
        :return processed image
        """
        for prep_params in self.config.preprocessing:
            image = PREPROC_FNS[prep_params.type](image, prep_params)
        return image


# Custom implementation of classification accuracy metric.
class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """ Resets collected matches """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {'direction': 'higher-better',
                             'type': 'accuracy'}}


def resize(image, params):
    shape = params['height'], params['width']
    return cv2_resize(image, shape)


def get_torch_dataloader(folder, transform, batch_size=1):
    val_dataset = datasets.ImageFolder(root=folder, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )
    return val_loader


def crop(image, params):

    height, width = image.shape[:2]

    dst_height = int(height * params['central_fraction'])
    dst_width = int(width * params['central_fraction'])

    if height < dst_height or width < dst_width:
        resized = np.array([width, height])
        if width < dst_width:
            resized *= dst_width / width
        if height < dst_height:
            resized *= dst_height / height
        image = cv2_resize(image, tuple(np.ceil(resized).astype(int)))

    top_left_y = (height - dst_height) // 2
    top_left_x = (width - dst_width) // 2
    return image[top_left_y:top_left_y + dst_height, top_left_x:top_left_x + dst_width]


PREPROC_FNS = {'resize': resize, 'crop': crop}


def get_configs(args):
    if not args.weights:
        args.weights = '{}.bin'.format(os.path.splitext(args.model)[0])

    if not args.weights_fp8:
        args.weights_fp8 = '{}.bin'.format(os.path.splitext(args.model_fp8)[0])

    model_config_fp32 = {
        'model_name': 'sample_model',
        'model': os.path.expanduser(args.model),
        'weights': os.path.expanduser(args.weights)
    }

    model_config_fp8 = {
        'model_name': 'sample_model',
        'model': os.path.expanduser(args.model_fp8),
        'weights': os.path.expanduser(args.weights_fp8)
    }

    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 4,
        'eval_requests_number': 4
    }
    dataset_config = {
        'data_source': os.path.expanduser(args.dataset),
        #'annotation_file': os.path.expanduser(args.annotation_file),
        'has_background': True,
        #'has_background': False,
        'preprocessing': [
            {
                'type': 'crop',
                'central_fraction': 0.875
            },
            {
                'type': 'resize',
                'width': 224,
                'height': 224
            }
        ],
    }

    return model_config_fp32, model_config_fp8, engine_config, dataset_config


def relative_change(x, y):
    num = np.abs(x - y)
    denum = np.maximum(np.abs(x), np.abs(y))
    return np.mean(num / (denum + 1e-10))

def read_models(args):
    model_config_fp32, model_config_fp8, engine_config, dataset_config = get_configs(args)

    # Step 1: Load the model.
    model_fp32 = ie.read_model(model_config_fp32['model'])
    model_fp8 = ie.read_model(model_config_fp8['model'])

    return model_fp32, model_fp8


def main():
    argparser = get_common_argparser()
    # argparser.add_argument(
    #     '-d',
    #     '--data_dir',
    #     help='Imagenet directory',
    #     required=True
    # )
    argparser.add_argument(
        '-m_fp8',
        '--model_fp8',
        help='Path to the xml file with model',
        required=True)

    argparser.add_argument(
        '-w_fp8',
        '--weights_fp8',
        default=None,
        help='Path to the bin file with model',
        required=False)
    
    argparser.add_argument(
        '-m_n',
        '--model_name',
        help='timm model name',
        required=True)


    # Steps 1-7: Model optimization
    args = argparser.parse_args()
    model_fp32, model_fp8 = read_models(args)

    dst_dir = args.model_name + '_data_'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    model = create_timm_model(args.model_name)
    model.eval().cpu()
    transform = get_model_transform(model)

    batch_one_dataloader = get_torch_dataloader(args.dataset, transform, batch_size=1)

    #cmodel_fp32, cmodel_fp8 = ie.compile_model(model_fp32, 'CPU'), ie.compile_model(model_fp8, 'CPU')

    ops_fp32 = model_fp32.get_ordered_ops()
    ops_fp8 = model_fp8.get_ordered_ops()
    layer_names = []

    for op in ops_fp32:
        l_name = op.friendly_name
        try:
            model_fp32.add_outputs([op.output(0)])#[l_name])
        except:
            continue
        layer_names.append(l_name + ':0')

    for op in ops_fp8:
        l_name = op.friendly_name

        try:
            model_fp8.add_outputs([op.output(0)])#[l_name])
        except:
            print("Can not add layer to output fp8 : ", l_name)
            continue
        print("Dump fp8 : ", l_name)


    cmodel_fp32, cmodel_fp8 = ie.compile_model(model_fp32, 'CPU'), ie.compile_model(model_fp8, 'CPU')

    #save_model(cmodel_fp8, os.path.join(os.path.curdir, 'optimized_add_output'))

    max_diff  = {}
    mean_diff = {}

    n = 1
    for im, _ in batch_one_dataloader:
        if n > 10:
            break
        results_fp32 = cmodel_fp32.infer_new_request({0: im})
        results_fp8 = cmodel_fp8.infer_new_request({0: im})

        results_fp32_v = iter(results_fp32.values())
        results_fp32_k = iter(results_fp32.keys())

        results_fp8_v = iter(results_fp8.values())
        results_fp8_k = iter(results_fp8.keys())

        res_fp32 = {}
        res_fp8 = {}

        for k32 in results_fp32_k:
            v32 = next(results_fp32_v)
            try:
                res_fp32[k32.any_name] = v32
            except:
                pass
        
        bad_idx = 0
        for k8 in results_fp8_k:
            v8 = next(results_fp8_v)
            try:
                res_fp8[k8.any_name] = v8
            except:
                res_fp8["bad_value_{}".format(bad_idx)] = v8
                bad_idx += 1
                pass

        for k in res_fp8.keys():
            np.save(f"{dst_dir}/{k.replace('/', '_')}_8.npy", res_fp8[k])

        # for k in res_fp32.keys():
        #     if True or res_fp32[k].shape == (1, 32, 112, 112):
        #         np.save(f"{dst_dir}/{k.replace('/', '_')}_32.npy", res_fp32[k])

        for k in res_fp32.keys():
            if not k in res_fp8:
                print("Skip layer diff: ", k)
                continue
            data_fp32 = res_fp32[k].flatten()
            data_fp8 = res_fp8[k].flatten()
            diff = relative_change(data_fp8, data_fp32)
            if k in mean_diff:
                mean_diff[k] = ((n - 1) * mean_diff[k] + diff) / n
            else:
                mean_diff[k] = diff

            if k in max_diff:
                if diff > max_diff[k]:
                    max_diff[k] = diff
                    np.save(f"{dst_dir}/{k.replace('/', '_')}_32.npy", res_fp32[k])
                    np.save(f"{dst_dir}/{k.replace('/', '_')}_8.npy", res_fp8[k])
            else:
                max_diff[k] = diff
                np.save(f"{dst_dir}/{k.replace('/', '_')}_32.npy", res_fp32[k])
                np.save(f"{dst_dir}/{k.replace('/', '_')}_8.npy", res_fp8[k])
        
        n += 1
        break

    # for k, v in mean_diff:
    #     print

    d_view = [ (v,k) for k, v in mean_diff.items() ]
    d_view.sort(reverse=True) # natively sort tuples by first element
    for v, k in d_view:
        print("Diff {}: {}".format(k, v))




if __name__ == '__main__':
    main()

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import onnx

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    #model.cuda(config.gpus[0])

    model.eval()

    x = torch.rand(1, 3, 1024, 1024) # eg. torch.rand([1, 3, 256, 256])
    _ = model(x)

    torch.onnx.export(model,
                x,  # model input
                'convnext_cls_potsdam.onnx',  # where to save the model
                export_params=True,
                opset_version=14,
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=False)
    
    model = onnx.load('./convnext_cls_potsdam.onnx')

    class_names = {
        0: 'ImSurf', 
        1: 'Building', 
        2: 'LowVeg', 
        3: 'Tree', 
        4: 'Car', 
        5: 'Clutter'
    }

    m = model.metadata_props.add()
    m.key = 'model_type'
    m.value = json.dumps('Segmentor')


    m = model.metadata_props.add()
    m.key = 'class_names'
    m.value = json.dumps(class_names)

    m = model.metadata_props.add()
    m.key = 'resolution'
    m.value = json.dumps(9)

    m = model.metadata_props.add()
    m.key = 'tiles_size'
    m.value = json.dumps(1024)

    m = model.metadata_props.add()
    m.key = 'tiles_overlap'
    m.value = json.dumps(15)

    m = model.metadata_props.add()
    m.key = 'seg_thresh'
    m.value = json.dumps(0.5)

    m = model.metadata_props.add()
    m.key = 'seg_small_segment'
    m.value = json.dumps(11)


    onnx.save(model, './convnext_cls_potsdam.onnx')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler import Evaler
from yolov6.utils.general import increment_name


@torch.no_grad()
def run(
    data: str,
    weights: str = None,
    batch_size: int = 32,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    task: str = "val",
    device: str = "",
    half: bool = False,
    save_dir: str = "",
    name: str = "",
    dataloader=None,
    model=None,
):
    """Run the evaluation process

    This function is the main process of evaluation, supporting image file and
    dir containing images. It has tasks of 'val', 'train' and 'speed'. Task
    'train' processes the evaluation during training phase. Task 'val'
    processes the evaluation purely and return the mAP of model.pt. Task
    'speed' precesses the evaluation of inference speed of model.pt.
    """

    # task
    Evaler.check_task(task)  # assertions
    save_directory = increment_name(Path(save_dir) / name)
    save_directory.mkdir(parents=True, exist_ok=True)

    # reload thres/device/half/data according task
    conf_thres, iou_thres = Evaler.reload_thres(conf_thres, iou_thres, task)
    torch_device = Evaler.reload_device(device, model, task)
    half = torch_device.type != "cpu" and half
    data = Evaler.reload_dataset(data)

    # init
    val = Evaler(
        data,
        batch_size,
        img_size,
        conf_thres,
        iou_thres,
        torch_device,
        half,
        save_directory,
    )
    model = val.init_model(model, weights, task)
    dataloader = val.init_data(dataloader, task)

    # eval
    model.eval()
    pred_result = val.predict_model(model, dataloader, task)
    eval_result = val.eval_model(pred_result, model, dataloader, task)
    return eval_result


def main(args):
    run(
        args["data"],
        args["weights"],
        args["batch_size"],
        args["img_size"],
        args["conf_thres"],
        args["iou_thres"],
        args["task"],
        args["device"],
        args["half"],
        args["save_dir"],
        args["name"],
        dataloader=None,
        model=None,
    )


if __name__ == "__main__":
    with open(ROOT / "yolov6.yml") as infile:
        configs = yaml.safe_load(infile.read())

    print(configs)
    main(configs)

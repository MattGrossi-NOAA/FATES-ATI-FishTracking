<summary>Install</summary>
This work is based on YOLOv8. Follow the following instructions to install YOLOv8. <br> 
Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart).

</details>
<summary>How to run the code</summary>

```bash
cd  ultralytics
#multi GPU detection
 python -m torch.distributed.launch --nproc_per_node 2 main_det.py --batch-size 64 --data coco.yaml --weights yolov5s.pt
```
In the above detection training, 2 GPUs are used. To change the number of GPUS, modify the devices in the main_det.py code. <br> 
Similarly, run the evaluation and tracking code.

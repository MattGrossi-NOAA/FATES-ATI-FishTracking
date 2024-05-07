# Multi-fish Tracking for Marine Biodiversity Monitoring
### Abstract
Accurate recognition of multiple fish species is essential in marine ecology and fisheries. Precisely classifying and tracking these species enriches our comprehension of their movement patterns and empowers us to create precise maps of species-specific territories. Such profound insights are pivotal in conserving endangered species, promoting sustainable fishing practices, and preserving marine ecosystems' overall health and equilibrium.  
To partially address these needs, we present a proposed model that combines YOLOv8 for object detection with ByteTrack for tracking. YOLOv8's oriented bounding boxes help to improve object detection across angles, while ByteTrack's robustness in various scenarios makes it ideal for real-time tracking. Experimental results using the SEAMAPD21 dataset show the model's effectiveness, with YOLOv8n being the lightweight yet modestly accurate option, suitable for constrained environments. The study also identifies challenges in fish tracking, such as lighting variations and fish appearance changes, and proposes solutions for future research. Overall, the proposed model shows promising fish tracking and counting results, which is essential for monitoring marine life.


## Citation
If you find this work useful, please cite:
```bib
@inproceedings{alaba2024multi,
  title={Multi-fish Tracking for Marine Biodiversity Monitoring},
  author={Alaba, Simegnew and Prior, Jack and Shah, Chiranjibi and Nabi, MM and Ball, John and Moorhead, Robert and Han, Deok and  Campbell, Matthew and Wallace, Farron and  Grossi, Matthew D. },
  journal={Ocean Sensing and Monitoring XV},
  year={2024}
}
```

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

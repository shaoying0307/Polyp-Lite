# Polyp-Lite: Bridging the Efficiency–Accuracy Gap in Lightweight Polyp Detection Scenarios via Dynamic Feature Aggregation
Although colonoscopy is the cornerstone of colorectal cancer (CRC) screening, manual diagnosis remains severely constrained by high missed detection rates and limited healthcare resources. While deep learning offers a novel method for addressing these concerns, the existing models often fall into the pseudolightweight trap—reducing parameters without similarly reducing computational demands—making them ill-suited for the stringent constraints of clinical edge devices. To address this computational bottleneck, a Polyp-Lite polyp detection architecture is proposed in this study. This solution involves a heterogeneous dual-branch backbone (DualFusionNet) that integrates dynamic kernel convolution with a local-global fusion module (LGFM) to decouple spatial details from deep semantic information. Coupled with an encoder with a multifeature fusion block (MFFB) as its core feature fusion unit and a light decoder, the model further eliminates computational redundancy. Experimental results demonstrate that compared with the mainstream state-of-the-art (SOTA) models, Polyp-Lite maintains high accuracy while substantially reducing computational loads, requiring only 1.6 GFLOPs. Consequently, this algorithm is an ideal choice for battery-powered, resource-constrained edge devices, establishing significant energy efficiency advantages for the clinically challenging detection of small targets. This breakthrough effectively mitigates the tradeoff between accuracy and efficiency, paving the way for a lightweight pathway with high clinical translatability for allowing high-performance auxiliary diagnostics in endoscopic systems.
<p align="center">
  <img src="images/1.png" width="70%">
</p>  


## Dataset  

We construct a large-scale hybrid polyp dataset that integrates major public benchmark datasets, including [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/), [CVC-ColonDB](https://www.kaggle.com/datasets/longvil/cvc-colondb), [ETIS-Larib](https://polyp.grand-challenge.org/ETISLarib/), [Kvasir-SEG](https://datasets.simula.no/kvasir-seg), [PolypGen](https://github.com/DebeshJha/PolypGen), and [PolypDB](https://github.com/DebeshJha/PolypDB?tab=readme-ov-file), comprising more than 15,000 high-quality endoscopic images. The dataset under consideration spans a wide range of intestinal environments, lighting conditions, and polyp morphologies, providing rich, representative samples for model training. To ensure that the evaluations are clinically valid and to prevent sequence data leakage, patient-level partitioning principles are strictly followed throughout. All images derived from a given video sequence or patient must be assigned to the same subset, thereby ensuring that the tested models are evaluated on their generalizability to unseen lesions rather than on their “memory” of similar frames.


## How to Use  
### Requirements  
#### Note  

The code for this model requires execution within the mmdetection environment, thus necessitating the prior installation of the requisite runtime environment. The installation repository is located at https://github.com/open-mmlab/mmdetection, with installation instructions available at https://mmdetection.readthedocs.io/en/latest/get_started.html.  
The model's source code is housed within the Polyp-Lite folder. The configuration file path is located at [configs/my_model.py](configs/my_model-polyp.py)


1.Train  

`python tools/train.py configs/my_model/<your-config-file>`  

2.Test  

`python tools/test.py <your-config-file> <your-model-weights-file> --out <save-pickle-path>`  

## Experiments Results  
### Comparison with SOTA and Lightweight Models  

| SOTA Models | mAP50:95 (%) | Params (M) | FLOPs (G) | Model Size (MB) |
|---|---:|---:|---:|---:|
|   **One-Stage**  |
| YOLOv8n | 78.80 | 2.69 | 6.8 | 5.4 |
| YOLOv10n  | 78.76 | 2.27 | 6.5 | 5.5 |
| YOLOv12n | 75.89 | 2.45 | 6.1 | 6.1 |
|   **Two-Stage**   |
| Faster R-CNN | 70.70 | 41.35 | 214.0 | 83.4 |
| Cascade R-CNN | 72.30 | 69.16 | 186.2 | 125.7 |
|  **End-to-End**  |
| DETR-R50 | 71.13 | 41.56 | 85.74 | 81.4 |
| RT-DETR-R18 | 76.24 | 19.88 | 56.9 | 38.6 |
|  **Lightweight Models**  |
| LeYOLO-Nano | 67.95 | 1.09 | 2.5 | 2.7 |
| CE-YOLO-Tiny | 70.23 | 4.16 | 2.1 | 3.5 |
| CRH-YOLO | 68.52 | 0.91 | 11.3 | 2.0 |
| Polyp-DeNet | 76.85 | 3.7 | 7.8 | 3.7 |
| EfficientDet-D3 | 57.82 | 11.91 | 23.1 | 21.6 |
| SSD-Lite | 51.61 | 3.03 | 0.7 | 5.0 |
|  **Our Model**   |
| **Polyp-Lite** | 75.18 | 0.90 | 1.6 | 2.5 |

### Generalisation Ability  
In this study, the generalizability of the proposed model to pathological tissue features is further evaluated. This evaluation is based on the binary classification dataset provided by [Li et al.](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR), which includes adenomatous and hyperplastic polyps. As demonstrated in Table 5, Polyp-Lite achieves mAP50:95 scores of 84.05% and 84.20% for the two subtypes, respectively. Despite marginal discrepancies of 2–3% relative to the baseline model and YOLOv8n, given minimal parameter count of 0.90 M and the computational overhead of 1.6 GFLOPs for Polyp-Lite, it nevertheless exhibits remarkable performance under extreme lightweight constraints. The model is shown to effectively extract key morphological features for distinguishing high-risk adenomas. These findings suggest that it has potential practical value in clinical pathological diagnosis scenarios. 

| MODEL      | Adenomatous (mAP50:95 %) | Hyperplastic (mAP50:95 %) | Params (M) | FLOPs (G) |  FPS  |
|------------|--------------------------:|---------------------------:|-----------:|----------:|------:|
| POLYP-LITE |                     84.05 |                      84.20 |       0.90 |       1.6 | 105.8 |
| Baseline   |                     86.12 |                      86.52 |       2.21 |       6.0 | 129.5 |
| YOLOv8n    |                     86.86 |                      87.52 |       2.69 |       6.8 | 114.9 |

### Visualisation  
The qualitative detection performance attained across the four complex scenarios is shown in Figure 16. When confronted with low contrast levels and poorly defined occlusions, Polyp-Lite leverages the enhanced ability of DualFusionNet to perceive subtle textures and edges to overcome the failure of the baseline YOLOv8n model to detect microlesions that are not clearly visible. In rigorous artifact interference tests, the global contextual information provided by the semantic branch effectively suppresses background noise, thereby preventing false positives from being generated by the baseline model. In scenarios involving dense, multiple polyps, this model successfully recovers minute lesions that the baseline model misses because of its advanced boundary regression capabilities. This fully validates its resistance to interference and its robustness in complex endoscopic environments.  

<p align="center">
  <img src="images/2.png" width="80%">
</p> 


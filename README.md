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
The model's source code is housed within the Polyp-Lite folder. The configuration file path is located at [configs/liberrynet-wildberry.py](https://github.com/Leo-shaoying/LiBerryNet/blob/main/liberrynet/configs/liberrynet-wildberry.py)


1.Train  

`python tools/train.py configs/my_model/<your-config-file>`  

2.Test  

`python tools/test.py <your-config-file> <your-model-weights-file> --out <save-pickle-path>`  

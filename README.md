# DASH-DA: A new Adversarial Disentanglement Framework for Semi-Supervised Heterogeneous Domain Adaptation

## Framework overview 
![DASH-DA architecture](Arch.jpg)

**Considered setting:** The key challenge in Semi-Supervised Domain Adaptation lies in transferring knowledge from a source domain with abundant labeled data to a target domain with scarce labeled and abundant unlabeled data. This task is challenging due to potential distribution shifts, but the complexity further increases when source and target data differ in modality representation, such as when they are captured using different sensors (e.g., RGB and thermal cameras). This scenario, known as Semi-Supervised Heterogeneous Domain Adaptation (SSHDA), introduces an even more severe distribution shift due to modality differences across domains.

**Proposed approach:** To address these challenges, we propose DASH-DA (Disentangled Adversarial Semi-supervised Heterogeneous Domain Adaptation), an end-to-end neural framework designed to disentangle domain-invariant features, which are essential for the downstream task, from domain-specific features that may hinder cross-modality transfer. Additionally, to further strengthen the domain invariance of the extracted features, we incorporate an adversarial learning term based on Wasserstein distance. We evaluate our framework on three multi-modal benchmarks: SUN RGB-D, TRISTAR, and HANDS. Extensive experiments demonstrate that DASH-DA outperforms both baselines and state-of-the-art (SoTA) SSHDA approaches in challenging scenarios featured by minimal label supervision on the target data, across diverse backbones architectures.


## Code organization

Train and test of the proposed framework are performed in file `main.py`.
Our framework has been implemented using two different backbone architectures: *ResNet-18* (`backbone_resnet.py`) and *TinyViT* (`backbone_vit.py`). Auxiliary functions are provided in `functions.py`, while parameters and hyperparameters are stored in `param.py`.

### Data
To prepare the data for use with our framework, first download the original datasets and normalize them according to your application requirements. Once preprocessed, place the dataset (including both modality files and the label file) in a folder named ./Datasets/dataset_name.

Finally, run `preproc.py` to generate the necessary data files.

### Input arguments 
Scripts take the following input arguments in order (they are used for data loading configuration and may be modified to meet your own dataset conventions):
1) **Dataset Name**: Choose from *SUNRGBD*, *TRISTAR*, or *HANDS*.  
2) **Source Data Prefix**: Specifies the modality used as the source domain. Options include *RGB* or *DEPTH* (for *SUNRGBD* and *HANDS*) and *THERMAL* or *DEPTH* (for *TRISTAR*). The remaining modality is automatically assigned as the target domain.  
3) **Backbone Architecture**: Select either *ResNet-18* or *TinyViT*.  
4) **GPU Number**: Specify the GPU to use.  
5) **Labeled Target Samples**: Define the number of labeled target samples (e.g., 5, 10, 25, 50) or provide multiple values.  
6) **Split Number**: Choose a specific train-test split (e.g., 0, 1, 2, 3, 4) or select multiple splits.

Example of running istruction:

<!---->

    python main.py -d SUNRGBD -s DEPTH -b ResNet-18 -n_gpu 1 -ns 5 10 50 -np 1 2 5


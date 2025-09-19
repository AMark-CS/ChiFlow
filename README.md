# Torus Flow Matching for Protein Backbone Generation


**Torus Flow Matching** is a generative model for protein design that focuses on dihedral angles, leveraging the toroidal geometry of these angles. This approach is part of the broader **FoldFlow** framework and introduces a specialized flow matching technique tailored for the torus manifold. For more information, please contact hhhzhchhh939@gmail.com

# Installation
To use the Torus Flow Matching model, follow these steps to set up the environment:
python=3.10
```bash
git clone https://github.com/AMark-CS/ChiFlow.git
cd ChiFlow
conda env create -f environment.yaml
conda activate chiflow
pip install -e .
```

# Training

## Training Torus Flow Matching on a Single Protein
To quickly test the Torus Flow Matching model, you can train it on a single protein. This example uses the `2f60` protein:

```bash
python runner/train.py local=example model_type=torus
```

This should converge in approximately 10-20 minutes on a V100 GPU.

## Full Dataset Training
For training on the full dataset, ensure the dataset is preprocessed and available. Update the configuration file to specify the dataset paths:

```yaml
model_type: torus
data:
  cluster_path: ./data/processed_pdb/clusters-by-entity-30.txt
```

Then, run the training script:

```bash
python runner/train.py model_type=torus
```

# Inference

To perform inference using the Torus Flow Matching model, specify the checkpoint path in the configuration file:

```yaml
inference:
  model_type: torus
  weights_path: path/to/torus_checkpoint.pth
```

Run the inference script:

```bash
python runner/inference.py
```

You can also override configurations directly from the command line:

```bash
python runner/inference.py inference.weights_path=path/to/new_ckpt.pth inference.model_type=torus
```

# Citation

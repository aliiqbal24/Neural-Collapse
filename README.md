# Neural Collapse Replication

This repository contains a paper presentation and replication project on
**Neural Collapse**, a phenomenon where deep classifier features and final-layer
weights converge to a highly structured geometry near the end of training.

It includes replication code, as well as my more technical report and presentation slides

The project focuses on reproducing Neural Collapse behavior on MNIST with a
small ResNet-18 setup. The notebook trains the model with either cross-entropy
or MSE loss, then tracks the standard NC1-NC4 measurements and related MSE loss
decomposition during training.

## Repository Contents

- `neuralcollapse.ipynb` - main replication notebook with training, analysis,
  and plotting code.
- `ProjectReport.pdf` - written project report summarizing the paper,
  replication setup, results, and discussion.
- `Presentation.pptx` - slide deck for the paper presentation.

## Running the Replication

Open `neuralcollapse.ipynb` in Google Colab or Jupyter. A GPU runtime is
recommended but not needed for the full experiment.

The notebook downloads MNIST automatically and supports two modes:

- `debug = True` for a quick test run.
- `debug = False` for the full training run.

To compare objectives, set `loss_name` to either:

- `MSELoss`
- `CrossEntropyLoss`

## Requirements

The notebook uses Python 3 with:

- PyTorch
- torchvision
- NumPy
- SciPy
- Matplotlib
- tqdm

## References

The replication is based on the Neural Collapse papers cited in the notebook,
including Papyan, Han, and Donoho's original Neural Collapse work

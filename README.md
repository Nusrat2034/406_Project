# Topic 30: FGSM / PGD Adversarial Examples on an Image Classifier

CSE406 Project 2026. Reference papers: Goodfellow et al. 2015 (`1412.6572v3.pdf`,
FGSM), Madry et al. 2018 (`1706.06083v4.pdf`, PGD + adversarial training),
Xu et al. 2017 (`1704.01155v2.pdf`, feature squeezing defense). Full task
description in `CSE406ProjectJan2026.pdf`.

Rule: the attack (and defense) logic must be implemented from scratch --
standard frameworks (PyTorch, torchvision) are allowed, but ready-made
attack/defense libraries (foolbox, ART, CleverHans) are not.

## Experiment design

A 2x2 grid: dataset complexity (MNIST vs CIFAR-10) x model complexity
(a small custom CNN vs ResNet18), to see how each factor affects
adversarial robustness.

| | Simple CNN | ResNet18 |
|---|---|---|
| MNIST | `step1_train_classifier.py` + `step2_fgsm_attack.py` + `step3_pgd_attack.py` | `resnet18_mnist_cifar10.ipynb` (Part A) |
| CIFAR-10 | `cifar10_fgsm_pgd.ipynb` | `resnet18_mnist_cifar10.ipynb` (Part B) |

## Results (clean vs FGSM vs PGD accuracy)

| Pair | Clean acc | Params | Results CSV |
|---|---|---|---|
| MNIST + Simple CNN | 98.84% | ~421K | `mnist_fgsm_pgd_results.csv` |
| CIFAR-10 + Simple CNN | 73.43% | ~620K | `cifar10_fgsm_pgd_results.csv` |
| MNIST + ResNet18 | 99.12% | ~11.17M | `resnet18_mnist_fgsm_pgd_results.csv` |
| CIFAR-10 + ResNet18 | 81.51% | ~11.17M | `resnet18_cifar10_fgsm_pgd_results.csv` |

Each CSV has one row per epsilon with `fgsm_accuracy` and `pgd_accuracy`
columns. Matching `*_clean.png` / `*_adversarial.png` files are sample
before/after images for the report.

Headline finding: CIFAR-10 is far more fragile than MNIST (accuracy collapses
at much smaller epsilon), and PGD is consistently stronger than FGSM at the
same epsilon.

## How to run

- `step1_train_classifier.py`, `step2_fgsm_attack.py`, `step3_pgd_attack.py`:
  plain Python, run locally (`python step1_train_classifier.py`, etc. --
  each later step loads the `.pt` saved by the previous one).
- `cifar10_fgsm_pgd.ipynb`, `resnet18_mnist_cifar10.ipynb`: upload to
  Google Colab or Kaggle (GPU runtime), run all cells. Both detect the
  platform automatically for downloading output files.

All training/attack code keeps pixels in `[0, 1]` (no mean/std
normalization), so `epsilon` has the same simple meaning everywhere.

## What's left

- **Defense mechanism (bonus, 10%)** -- not implemented yet. The two
  candidate approaches from the reference papers:
  - **Adversarial training** (Madry et al.): during training, replace/mix in
    adversarial examples generated with `fgsm_attack()` / `pgd_attack()`
    (already implemented in `step2_fgsm_attack.py` / `step3_pgd_attack.py`
    and both notebooks -- reuse these, don't reimplement).
  - **Feature squeezing** (Xu et al.): reduce color bit-depth / apply
    spatial smoothing and compare model predictions to detect adversarial
    inputs.
  Use the already-trained `.pt` models and the CSVs above to decide which
  epsilon range is worth targeting.
- Design Report (topology/timing diagrams, justification).

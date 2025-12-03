CV Classification Starter (PyTorch)
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

> *This is the starter code for Track A (CV • Image Classification) of the Deep Learning Fil Rouge project. It uses PyTorch and torchvision to build, train, and evaluate an image classification model on CIFAR-10.*
## 1) Choose your workflow
Pick **one** of the two paths below. Both produce the same outputs.

### A. Notebook-first (Colab or Kaggle)
1. Open `notebooks/00_colab_quickstart.ipynb` in Colab/Kaggle.
2. Switch the runtime to **GPU (T4)**  
   - Colab: `Runtime → Change runtime type → GPU → Save`.  
   - Kaggle: gear icon → enable **Accelerator** → select `T4 x1`.
3. Upload or clone this `cv-project` folder so the notebook sees `/content/cv-project` (Colab) or the working dir (Kaggle).
4. Run the notebook cells from top to bottom:
   - GPU check (`nvidia-smi`) confirms acceleration.
   - Dependency cell installs everything from `requirements.txt`.
   - Smoke test cell performs a one-batch forward/backward pass and writes `outputs/smoke_metrics.json`.
5. Choose your stopping point:
   - **Smoke test only:** finish after the smoke cell if you just need an environment check.
   - **Full run inside the notebook:** execute the two “Full run” cells at the end. They call the same commands as the CLI section below, producing `best.pt`, metrics files, and plots. Once those cells succeed you can skip the CLI instructions entirely.

### B. Local CLI
Create and activate a reusable conda environment (Python 3.10 recommended), then install the requirements:
```bash
conda create -n dl_project python=3.10 -y
conda activate dl_project
pip install -r requirements.txt
```
If you prefer to keep the environment ephemeral, you can also run commands via `conda run -n dl_project <command>`.

> **Keeping dependencies fresh:** whenever `requirements.txt` changes, run  
> `conda run -n dl_project pip install -r requirements.txt`  
> to install the newest packages (e.g., seaborn / scikit-learn for enhanced evaluation plots).

## 2) Train
```bash
python src/train.py --config configs/cv_cifar10.yaml
```
Artifacts are written to `outputs/`: `best.pt`, `log.csv`, `metrics.json`.

## 3) Evaluate
```bash
python src/evaluate.py --config configs/cv_cifar10.yaml --ckpt outputs/best.pt
```
Produces:
- `eval.json` with accuracy and macro/weighted precision/recall/F1.
- `per_class_metrics.csv` (precision/recall/F1/support per class).
- `confusion_matrix.png` (seaborn heatmap).
- `leaderboard.png` (per-class precision/recall/F1 bar chart).

## 4) Switch to your own images
- Put class subfolders under `data/` like `data/cats/`, `data/dogs/`, …
- In `configs/cv_cifar10.yaml`, set:
```yaml
data:
  dataset: imagefolder
  root: ./data
  val_split: 0.1
```

## 5) Quick FAQ (read this before asking for help)
- **“Do I run both the notebook and the CLI?”** No—pick the path that suits you. Notebook users stop after the optional cells; CLI users can ignore the notebook.
- **“Evaluation crashed because `best.pt` is missing.”** Training must finish successfully first. Check `outputs/log.csv` for clues, then rerun `python src/train.py ...`.
- **“My metrics look low.”** Confirm you ran on GPU (training a ResNet on CPU is very slow and may early-stop). You can also increase `epochs` or unfreeze more layers in `configs/cv_cifar10.yaml`.
- **“Can I re-use this for another track?”** Yes! Copy the notebook, change the starter folder, update the install cell to match the new `requirements.txt`, and point the train/eval commands to the other track’s config.

---
# Appendix: Project Architecture
> The project follows this structure:

```
cv-project/
  configs/
    config.yaml               # dataset path, batch_size, lr, epochs, scheduler, seed
  data/                       # small samples or links
  src/
    data.py                   # Datasets/Dataloaders (vision, nlp, tabular variants)
    model.py                  # Model factory (resnet18, lstm, mlp, yolo wrapper)
    train.py                  # Train/validate loop (metrics, early stopping, ckpt)
    evaluate.py               # Final metrics, confusion matrix / mAP / plots
    utils.py                  # Seed, logger, checkpoint, LR scheduler
  notebooks/
    00_quickstart.ipynb       # smoke test: load data, 1 batch through model
    01_train.ipynb            # main training (calls src/train.py)
    02_eval_and_export.ipynb  # export weights, plots, demo snippets
  outputs/                    # checkpoints, logs, figures
  README.md                   # how-to + results table + ethical note
```

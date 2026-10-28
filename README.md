# Human Action Recognition with Transfer Learning

## Interactive Streamlit app

A companion application is available in [streamlit_app/app.py](streamlit_app/app.py).

```bash
python -m pip install -r streamlit_app/requirements.txt
python -m streamlit run streamlit_app/app.py
```

Image inference requires a compatible trained .keras checkpoint and a JSON list of class labels in output order. The original dataset/checkpoint is absent.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Community Cloud setup, validation coverage, and runtime requirements. A live deployment is not yet confirmed.


An image-classification learning project exploring human-action categories with a VGG16-based model.

## Why this project matters

Recognizing actions provides a practical setting for studying transfer learning, image preprocessing, and class-level evaluation. Applications such as activity monitoring require much stronger validation than a notebook experiment alone.

## Explore the project

- [Human-action-recognition_.ipynb](Human-action-recognition_.ipynb): implementation and saved outputs
- [requirements.txt](requirements.txt): Python dependencies

The existing model summary describes a VGG16 feature extractor and dense classification layers with 15 output classes.

## Run locally

```bash
git clone https://github.com/Arjun1443/Human-Action-Recognition.git
cd Human-Action-Recognition
python -m pip install -r requirements.txt
python -m pip install jupyter
python -m jupyter notebook
```

Review dataset paths in the notebook before execution. The image dataset is not included in the inspected repository tree. Obtain the appropriate dataset under its terms and update paths for your environment.

## Technical focus

Python · TensorFlow/Keras · transfer learning · OpenCV · image preprocessing · classification evaluation

## Evaluation and limitations

Recompute metrics on a clearly defined held-out split before citing performance. A frame-based classifier does not by itself establish temporal action understanding, real-time video support, or reliability for surveillance and healthcare use cases.

This repository is an educational model experiment, not a validated monitoring system.

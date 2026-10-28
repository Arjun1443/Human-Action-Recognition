import json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageOps
from ui import page

page('Human Action Studio','Explore a still-image action classifier with your trained checkpoint.','COMPUTER VISION')
REPO='Human-Action-Recognition'
DEFAULT_LABELS=None
st.caption('Checkpoint-based inference · Model-defined RGB images · original notebook preserved')
with st.sidebar:
    preprocessing=st.selectbox('Input preprocessing',['Scale pixels to 0–1','VGG16 preprocess_input','Raw 0–255 pixels'])
    st.caption('Select the same preprocessing used during training. Incorrect settings make predictions unreliable.')
upload=st.file_uploader('Upload an image',type=['png','jpg','jpeg'])
model_file=st.file_uploader('Trained Keras checkpoint (.keras)',type=['keras'])
label_file=st.file_uploader('Class labels as a JSON list (in output order)',type=['json'])
labels=DEFAULT_LABELS
if label_file:
    try:
        labels=json.load(label_file)
        if not isinstance(labels,list) or not labels or not all(isinstance(x,str) for x in labels): raise ValueError()
    except Exception:
        st.error('Labels must be a nonempty JSON list of strings.'); st.stop()
if upload:
    try:
        image=ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
        st.image(image,width=360)
    except Exception:
        st.error('Please upload a valid image.'); st.stop()
if not model_file:
    st.info('No trained checkpoint is bundled with this notebook repository. Upload your exported .keras model to enable genuine inference. No prediction is generated without a model.')
if st.button('Classify image',type='primary',disabled=not (upload and model_file and labels)):
    try:
        import tensorflow as tf
        with st.spinner('Loading checkpoint and running inference…'):
            with tempfile.TemporaryDirectory() as directory:
                file=Path(directory)/'model.keras'
                file.write_bytes(model_file.getvalue())
                model=tf.keras.models.load_model(file,compile=False,safe_mode=True)
            shape=model.input_shape
            if not isinstance(shape,tuple) or len(shape)!=4 or shape[-1]!=3 or not shape[1] or not shape[2]:
                raise ValueError('Expected a model with fixed RGB input shape (batch, height, width, 3).')
            if shape[1]>1024 or shape[2]>1024: raise ValueError('Input dimensions exceed the demo limit.')
            pixels=np.asarray(image.resize((shape[2],shape[1])),dtype=np.float32)[None,...]
            if preprocessing=='Scale pixels to 0–1': pixels=pixels/255.0
            elif preprocessing=='VGG16 preprocess_input': pixels=tf.keras.applications.vgg16.preprocess_input(pixels)
            scores=np.asarray(model.predict(pixels,verbose=0))
            if scores.shape!=(1,len(labels)) or not np.isfinite(scores).all():
                raise ValueError('Checkpoint output must match the supplied labels and contain finite values.')
            values=scores[0]
            if (values<0).any() or not np.isclose(values.sum(),1.0,atol=.01):
                st.warning('The output is not a normalized probability distribution. Values are shown as model scores.')
            ranked=pd.DataFrame({'Class':labels,'Model score':values}).sort_values('Model score',ascending=False)
            st.metric('Top-scoring class',ranked.iloc[0]['Class'])
            st.bar_chart(ranked.set_index('Class'))
            st.download_button('Download scores',ranked.to_csv(index=False),'class-scores.csv','text/csv')
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error('Checkpoint loading or inference failed. Use a compatible Keras model without custom code and matching preprocessing.')
with st.expander('Export and deployment requirements',expanded=True):
    st.code('model.save("model.keras")')
    st.write('Export from a completed training run. Upload only a checkpoint you trust. The app uses safe-mode loading and never accepts pickle artifacts. Large model uploads can exceed Community Cloud memory limits; use a compact model.')
    st.write('This app performs still-image classification. It does not claim temporal action recognition, medical reliability, or notebook benchmark reproduction.')
    if REPO=='CNN':
        st.write('A train_cloud_model.py script is included to train a compact CIFAR-10 baseline locally and export model.keras. The script downloads CIFAR-10 and uses a bounded training subset; report its own validation results separately.')

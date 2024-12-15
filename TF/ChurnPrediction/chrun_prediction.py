import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the trained model
model = load_model('cs_churn_tfmodel.keras')

# Print model input/output shapes
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)
model.summary()


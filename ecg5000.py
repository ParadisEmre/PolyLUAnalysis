# Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras import layers, losses 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import os
import math 

print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(f"GPU detected: {gpu}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU")

# Activation Functions Definitions
@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Sigmoid")
def F_Sigmoid(x):
    return tf.math.sigmoid(x)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Tanh")
def F_Tanh(x):
    return tf.math.tanh(x)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Softplus")
def F_Softplus(x):
    return tf.math.softplus(x)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_ChangedSoftplus")
def F_ChangedSoftplus(x):
    return tf.math.softplus(x) - tf.math.log(2.0)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_ReLU")
def F_ReLU(x):
    return tf.nn.relu(x)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_LeakyReLU")
def F_LeakyReLU(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_ELU")
def F_ELU(x):
    return tf.where(x >= 0.0, x, 1.0 * (tf.math.exp(x) - 1.0))

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Swish")
def F_Swish(x):
    return tf.nn.swish(x)

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Mish")
def F_Mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_PolyLU")
@tf.custom_gradient
def F_PolyLU(x):
    # forward pass nan safe
    positive_part = tf.nn.relu(x)
    negative_part = tf.where(
        x < 0.0,
        (1.0 / (1.0 - x)) - 1.0,
        0.0
    )
    y = positive_part + negative_part

    # backward pass nan safe
    def grad(dy):
        derivative = tf.where(
            x < 0.0,
            1.0 / tf.square(1.0 - x),
            1.0
        )
        
        return dy * derivative
    
    return y, grad

print("\nActivation functions are set.\n")

# Load ECG5000 Dataset from Tutorial URL
def load_ecg_dataset():
    csv_url = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'

    dataframe = pd.read_csv(csv_url, header=None)
    raw_data = dataframe.values
    print("Loaded data")

    # The last element contains the labels
    labels = raw_data[:, -1]

    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]

    # Train-test split (80-20) from tutorial
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )
    print(f"Data split: {len(train_data)} training, {len(test_data)} testing.")

    # Normalize 0 1
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    # Separate the normal rhythms from the abnormal rhythms.
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]

    x_test_full = test_data
    y_test_full = test_labels

    return normal_train_data, x_test_full, y_test_full

# Autoencoder model
def autoencoder(activation_function):

    model = keras.Sequential(
        layers=[
            # Encoder
            layers.Input(shape=(140,)),
            
            layers.Dense(32),
            layers.Activation(activation_function),
            
            layers.Dense(16),
            layers.Activation(activation_function),
            
            layers.Dense(8),
            layers.Activation(activation_function),
            
            # Decoder
            layers.Dense(16),
            layers.Activation(activation_function),
            
            layers.Dense(32),
            layers.Activation(activation_function),
            
            layers.Dense(140, activation="sigmoid")
        ]
    )
    
    model.compile(optimizer='adam', loss='mae')
    
    return model

# Parameters 
NUM_EPOCHS = 300     # Tutorial 20 epoch but used CIFAR-10 settings since its not specified
BATCH_SIZE = 256    # Tutorial 512 batch size but used CIFAR-10 settings since its not specified
NUM_RUNS = 5        # Paper 5 num runs
THRESHOLD = 1.0 # Tutorial 1.0 threshold

activations_to_test = {
    "ELU": F_ELU,
    "PolyLU": F_PolyLU,
    "Sigmoid": F_Sigmoid,
    "Tanh": F_Tanh,
    "ReLU": F_ReLU,
    "LeakyReLU": F_LeakyReLU,
    "Swish": F_Swish,
    "Mish": F_Mish,
}


# Load Dataset
normal_train_data, x_test_full, y_test_full = load_ecg_dataset()
end_result = {}
if normal_train_data is not None:
    print(f"(Settings: {NUM_EPOCHS} Batch Size: {BATCH_SIZE}, Runs:{NUM_RUNS} , Threshold:{THRESHOLD})\n")

    for name, activation_func in activations_to_test.items():
        print("\n-------------------------------------------------")
        print(f"TEST FOR: {name}")
        print("-------------------------------------------------")
        overall_accuracies = []
        overall_precisions = []
        overall_recalls = []

        for r in range(NUM_RUNS):
            print(f"Run {r+1}/{NUM_RUNS} starts")
            start_time_run = time.time()
            tf.keras.backend.clear_session()
            model = autoencoder(activation_func) 

            history = model.fit(normal_train_data, normal_train_data,
                                epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(x_test_full, x_test_full), 
                                shuffle=True,
                                callbacks=[keras.callbacks.EarlyStopping( #Some scenarios local min so nan fix (Emin olamıyorumn sebebinden)
                                    monitor='val_loss',
                                    patience=20,
                                    restore_best_weights=True
                                )],)

            reconstructions_train = model.predict(normal_train_data)
            train_loss = tf.keras.losses.mae(reconstructions_train, normal_train_data) 
            train_errors = train_loss.numpy()
            
            threshold = np.mean(train_loss) + np.std(train_loss)
            print(f"Threshold {threshold:.6f}")

            reconstructions_test = model.predict(x_test_full)
            test_loss = tf.keras.losses.mae(reconstructions_test, x_test_full) 
            test_errors = test_loss.numpy()

            preds_anomaly_bool = (test_errors < threshold)
            acc = accuracy_score(y_test_full, preds_anomaly_bool)
            prec = precision_score(y_test_full, preds_anomaly_bool, zero_division=0)
            rec = recall_score(y_test_full, preds_anomaly_bool, zero_division=0)

            overall_accuracies.append(acc)
            overall_precisions.append(prec)
            overall_recalls.append(rec)
            print(f"Run {r+1} ended. Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f} (Time: {time.time()-start_time_run:.1f}s)")

        avg_acc = np.mean(overall_accuracies) * 100
        avg_prec = np.mean(overall_precisions) * 100
        avg_rec = np.mean(overall_recalls) * 100

        end_result[name] = {
            "Average accuracy (%)": avg_acc,
            "Average precision (%)": avg_prec,
            "Average recall (%)": avg_rec
        }
        print(f"--- {name} Overall Result ({NUM_RUNS} runs) ---")
        print(f" Overall Accuracy: {avg_acc:.4f}%")
        print(f" Overall Precision: {avg_prec:.4f}%")
        print(f" Overall Recall: {avg_rec:.4f}%")

    print("\n=== END RESULTS ===")
    print(f"(Settings: {NUM_EPOCHS} Batch Size: {BATCH_SIZE}, Runs:{NUM_RUNS} , Threshold:{THRESHOLD})\n")
    results_df = pd.DataFrame.from_dict(end_result, orient='index')
    results_df = results_df[["Average accuracy (%)", "Average precision (%)", "Average recall (%)"]]
    results_df = results_df.sort_values(by="Average accuracy (%)", ascending=False)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df)

else:
    print("\nCould not load dataset.")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras import layers, losses 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import os
import math 

print(f"TensorFlow: {tf.__version__}")

# GPU
""" print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(f"GPU detected: {gpu}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU.")"""
    
# CPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices([], 'GPU')
        print("Forced CPU")
    else:
        print("No GPU so running on CPU")
except RuntimeError as e:
    print(e)

# Activation Functions Definitions CUSTOM IMPLEMENTATION
@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Sigmoid")
@tf.custom_gradient
def F_Sigmoid(x): 
    # forward pass
    y = 1.0 / (1.0 + tf.exp(-x))
    
    # backward pass
    def grad(dy):
        derivative = y * (1.0 - y)
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Tanh")
@tf.custom_gradient
def F_Tanh(x):
    # forward pass
    y = (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))
    
    # backward pass
    def grad(dy):
        derivative = 1.0 - tf.square(y)
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_ReLU")
@tf.custom_gradient
def F_ReLU(x):
    # forward pass
    y = tf.where(x < 0.0, 0.0, x)
    
    # backward pass
    def grad(dy):
        derivative = tf.where(x < 0.0, 0.0, 1.0)
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_LeakyReLU")
@tf.custom_gradient
def F_LeakyReLU(x):
    # forward pass
    y = tf.where(x < 0.0, 0.01 * x, x)
    
    # backward pass
    def grad(dy):
        derivative = tf.where(x < 0.0, 0.01, 1.0)
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_ELU")
@tf.custom_gradient
def F_ELU(x):
    # forward pass
    y = tf.where(x >= 0.0, x, 1.0  * (tf.exp(x) - 1.0))

    # backward pass
    def grad(dy):
        derivative = tf.where(x >= 0.0, 1.0, y + 1.0)
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_Swish")
@tf.custom_gradient
def F_Swish(x):
    # forward pass
    sigmoid_x = 1.0 / (1.0 + tf.exp(-x))
    y = x * sigmoid_x

    # backward pass
    def grad(dy):
        sigmoid_prime = sigmoid_x * (1.0 - sigmoid_x)
        derivative = sigmoid_x + x * sigmoid_prime
        return dy * derivative
    return y, grad

@tf.keras.utils.register_keras_serializable(package="Custom", name="F_PolyLU")
@tf.custom_gradient
def F_PolyLU(x):
    # forward pass nan safe
    positive_part = tf.where(x > 0.0, x, 0.0)
    negative_part = tf.where(x < 0.0, (1.0 / (1.0 - x)) - 1.0, 0.0)
    y = positive_part + negative_part
    
    # backward pass nan safe
    def grad(dy):
        derivative = tf.where(x < 0.0, 1.0 / tf.square(1.0 - x), 1.0)
        return dy * derivative
    
    return y, grad

print("\nCustom activation functions added.")

# Load Data
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalization (0-1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Channel Dimension Adding (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"x_train shape: {x_train.shape}") # (60000, 28, 28, 1)
    print(f"x_test shape: {x_test.shape}")   # (10000, 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)

# Create Model
def create_mnist_cnn_model(activation_function):
    model = keras.Sequential(
        layers=[
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(32, (3, 3), padding='same'),
            layers.Activation(activation_function),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.Activation(activation_function),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

NUM_EPOCHS = 1
BATCH_SIZE = 600 # Table 10 calculation
NUM_RUNS = 5

activations_to_test = {
    "PolyLU": F_PolyLU,
    "ELU": F_ELU,
    "Swish": F_Swish,
    "LeakyReLU": F_LeakyReLU,
    "ReLU": F_ReLU,
    "Tanh": F_Tanh,
    "Sigmoid": F_Sigmoid,
}


# Load Dataset
(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

all_results = {} 

if x_train is not None:
    print("\nMNIST EXPERIMENT RESULT\n")
    print(f"(Settings: Epoch: {NUM_EPOCHS} , Runs: {NUM_RUNS}, Batch Size: {BATCH_SIZE})\n")

    steps_per_epoch = len(x_train) // BATCH_SIZE
    print(f"Steps per epoch: {steps_per_epoch}")

    for name, activation_func in activations_to_test.items():
        print("\n-------------------------------------------------")
        print(f"TEST FOR: {name}")
        print("-------------------------------------------------")        
        run_times = []
        run_accuracies = []

        for r in range(NUM_RUNS):
            print(f"Run {r+1}/{NUM_RUNS} starts")
            start_time_run = time.time()

            tf.keras.backend.clear_session()
            model = create_mnist_cnn_model(activation_func)
            
            history = model.fit(x_train, y_train,
                                epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(x_test, y_test)) 

            end_time_run = time.time()
            elapsed_time = end_time_run - start_time_run
            run_times.append(elapsed_time)
            
            # Save the accuracy of last epoch to calculate average later
            final_val_acc = history.history['val_accuracy'][-1]
            run_accuracies.append(final_val_acc)

            print(f"Run {r+1} finished. Time: {elapsed_time:.2f}s, Val. Acc: {final_val_acc:.4f}")
        
        # Means
        avg_time_total = np.mean(run_times)
        avg_accuracy = np.mean(run_accuracies) * 100

        avg_time_per_epoch = avg_time_total / NUM_EPOCHS
        avg_step_per_sec = steps_per_epoch / avg_time_per_epoch

        all_results[name] = {
            "Evaluation accuracy (%)": avg_accuracy,
            "Step/sec": avg_step_per_sec,
            "Time/epoch (s)": avg_time_per_epoch
        }
        print(f"{name} Average Results ({NUM_RUNS} runs):\n")
        print(f"Avg. Accuracy: {avg_accuracy:.4f}%")
        print(f"Avg. Steps/sec: {avg_step_per_sec:.4f}")
        print(f"Avg. Time/epoch: {avg_time_per_epoch:.4f} s")

    print("\nMNIST EXPERIMENT RESULT\n")
    print(f"(Settings: Epoch: {NUM_EPOCHS} , Runs: {NUM_RUNS}, Batch Size: {BATCH_SIZE})\n")

    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df = results_df[["Evaluation accuracy (%)", "Step/sec", "Time/epoch (s)"]]
    results_df = results_df.sort_values(by="Time/epoch (s)", ascending=True)

    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df)

else:
    print("\nCould not load dataset.")
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import pandas as pd
import time
import os

print(f"TensorFlow: {tf.__version__}")
if tf.test.is_gpu_available():
    print("MUST RUN ON CPU")
else:
    print("Running on CPU")


def F_Sigmoid(x): 
    return 1.0 / (1.0 + tf.exp(-x))

def F_Tanh(x):
    return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

def F_ReLU(x):
    return tf.maximum(0.0, x)

def F_LeakyReLU(x):
    alpha = 0.01
    return tf.maximum(alpha * x, x)

def F_ELU(x):
    alpha = 1.0
    return tf.where(x >= 0.0, x, alpha * (tf.exp(x) - 1.0))

def F_Swish(x):
    sigmoid_x = 1.0 / (1.0 + tf.exp(-x))
    return x * sigmoid_x

# backward pass nan safe
@tf.RegisterGradient("PolyLU_Grad")
def _PolyLU_Grad(op, grad):
    x = op.inputs[0] 
    derivative = tf.where(
        x < 0.0,
        1.0 / tf.square(1.0 - x),
        tf.ones_like(x) 
    )
    return grad * derivative 
# forward pass nan safe
def F_PolyLU(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "PolyLU_Grad"}):
        positive_part = tf.maximum(0.0, x)     
        
        good_x = tf.where(x < 0.0, x, tf.zeros_like(x))
        negative_formula = (1.0 / (1.0 - good_x)) - 1.0
        negative_part = tf.where(x < 0.0, negative_formula, tf.zeros_like(x))
        
        y_forward = positive_part + negative_part
        y = tf.identity(y_forward)
        
    return y
print("\nCustom activation functions added.")

# Load Data
def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
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
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

NUM_EPOCHS = 5
BATCH_SIZE = 600 # Calculated by table 10 on the paper
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
            final_val_acc = history.history['val_acc'][-1]
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
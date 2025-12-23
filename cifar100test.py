# Library Imports
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import time
import sys
import keras.layers as layers

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

# Load Dataset
def load_cifar100_dataset():
    print("\nCIFAR-100 dataset loading...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    # Normalization to 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    print(f"x_train shape: {x_train.shape}") # (50000, 32, 32, 3)
    print(f"x_test shape: {x_test.shape}")   # (10000, 32, 32, 3)

    return (x_train, y_train), (x_test, y_test)


def create_cifar100_CNN_model(activation_function, batch_norm=True):
    model = keras.Sequential([keras.Input(shape=(32, 32, 3))])
    def add_bn_activation(layer):
        if batch_norm:
            layer.add(layers.BatchNormalization())
        layer.add(layers.Activation(activation_function))
        
    # Input --> (32x32x3)
    # Output --> (16x16x64)
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2)) # Dropout 1: 0.2
    # Input --> (16x16x64)
    #Output --> (8x8x128)
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3)) # Dropout 2: 0.3 
    #Input --> (8x8x128)
    #Output --> (4x4x256)
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4)) # Dropout 3: 0.4
    #Input --> (4x4x256)
    #Output --> (2x2x512)
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    add_bn_activation(model)
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5)) # Dropout 4: 0.5
    
    model.add(layers.Flatten()) # Output --> 2x2x512 = 2048

    # (1024)
    model.add(layers.Dense(1024))
    add_bn_activation(model)    
    # (512)
    model.add(layers.Dense(512))
    add_bn_activation(model)
    # (256)
    model.add(layers.Dense(256))
    add_bn_activation(model)
    # (128)
    model.add(layers.Dense(128))
    model.add(layers.Dropout(0.5)) # Last one was 0.5 so i guess its same.
    # (100)
    model.add(layers.Dense(100, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


NUM_EPOCHS = 500     # In paper its 500
NUM_RUNS = 5         # In paper its 5
BATCH_SIZE = 128    # In paper its 128

activations_to_test = {
    "PolyLU": F_PolyLU,
    "Sigmoid": F_Sigmoid,
    "Tanh": F_Tanh,
    "Softplus": F_Softplus,
    "ChangedSoftplus": F_ChangedSoftplus,
    "ReLU": F_ReLU,
    "LeakyReLU": F_LeakyReLU,
    "ELU": F_ELU,
    "Swish": F_Swish,
    "Mish": F_Mish,
}
activations_to_test["LeakyReLU"].__name__ = "LeakyReLU"
activations_to_test["ELU"].__name__ = "ELU"


(x_train, y_train), (x_test, y_test) = load_cifar100_dataset()


all_results = {}

print("\nCIFAR-100 EXPERIMENT")
print(f"(Settings: Epoch: {NUM_EPOCHS} , Runs: {NUM_RUNS}, Batch Size: {BATCH_SIZE})\n")

for name, activation_func in activations_to_test.items():
    print("\n-------------------------------------------------")
    print(f"TEST FOR: {name}")
    print("-------------------------------------------------")
    all_results[name] = {}
    
    for bn_state in [True, False]:
        bn = "With Batch Norm" if bn_state else "Without Batch Norm"
        print(f"\n{bn} test..")
        
        run_histories = []
        for r in range(NUM_RUNS):
            print(f"Iteration {r+1}/{NUM_RUNS}...")
            
            tf.keras.backend.clear_session()
            model = create_cifar100_CNN_model(activation_func, batch_norm=bn_state)
            
            start_time = time.time()
            
            history = model.fit(x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=NUM_EPOCHS,
                                validation_data=(x_test, y_test),
                                verbose=1) # İlerlemeyi görmek için 1
            
            end_time = time.time()
            print(f"Iteration {r+1} ended. {(end_time-start_time)/60:.2f} minutes")
            run_histories.append(history.history)

        
        # Average accuracy on training set at last epoch
        avg_train_acc = np.mean([h['accuracy'][-1] for h in run_histories])

        # Average validation accuracy at last epoch
        avg_val_acc = np.mean([h['val_accuracy'][-1] for h in run_histories])

        # Highest validation accuracy across all epochs
        max_val_acc = np.max([np.max(h['val_accuracy']) for h in run_histories])
        
        # Least validation accuracy across all epochs
        min_val_acc = np.min([np.min(h['val_accuracy']) for h in run_histories])


        all_results[name][bn] = {
            "Avg. Train Acc (Last Epoch)": avg_train_acc,
            "Avg. Val Acc (Last Epoch)": avg_val_acc,
            "Min. Val Acc (All Epochs)": min_val_acc,
            "Max. Val Acc (All Epochs)": max_val_acc,
        }
        print(f"--- {bn} test ended. ---")
        print(all_results[name][bn])


print("\nCIFAR-100 EXPERIMENT RESULT\n")
print(f"(Settings: Epoch: {NUM_EPOCHS} , Runs: {NUM_RUNS}, Batch Size: {BATCH_SIZE})\n")

for name, results_dict in all_results.items():
    print(f"{name}")
    for bn, metrics in results_dict.items():
        print(f"{bn}:")
        print(f"Avg. Train Acc: {metrics['Avg. Train Acc (Last Epoch)']:.4f}")
        print(f"Max. Test Acc  : {metrics['Max. Val Acc (All Epochs)']:.4f}")
        print(f"Avg. Test Acc  : {metrics['Avg. Val Acc (Last Epoch)']:.4f}")
        print("\n")
import mlflow
import mlflow.keras

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0  # Normalize the images
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)  # One-hot encode labels
test_labels = to_categorical(test_labels)

# Define your hyperparameter grid
i = 0
layers_grid = [2, 3, 4]
neurons_grid = [32, 64, 128]
activation_grid = ['relu', 'tanh']

best_performance = 0
best_hyperparameters = {}

mlflow.set_tracking_uri("file:./logs/")

def build_model(layers, neurons, activation):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten the MNIST images
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 classes for MNIST digits
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Grid search
for layers in layers_grid:
    for neurons in neurons_grid:
        for activation in activation_grid:
        
            i += 1
            print("\n ___ Architecture " , i , "___ \n" )
            model.summary()
        	
            with mlflow.start_run():
                mlflow.log_param("layers", layers)
                mlflow.log_param("neurons", neurons)
                mlflow.log_param("activation", activation)

                model = build_model(layers, neurons, activation)
                model.fit(train_images, train_labels, epochs=5, batch_size=32, verbose=1)
                performance = model.evaluate(test_images, test_labels, verbose=0)[1]

                mlflow.log_metric("accuracy", performance)

                # Optionally log the model
                mlflow.keras.log_model(model, "model")
                mlflow.log_artifact("model.h5", "model")
                

# Print best model's hyperparameters and performance
print("Best Hyperparameters:", best_hyperparameters)

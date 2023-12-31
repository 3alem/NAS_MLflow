import sys
import itertools
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


tf.random.set_seed(0)
mlflow.autolog()


# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0
# One-hot encode labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

input_shape =  (32, 32, 3)

num_classes = 10  # Assuming CIFAR-10 with 10 classes
aggregate_conf_matrix = np.zeros((num_classes, num_classes))

# Define your hyperparameter grid
i = 0

param_grid = {
    'layers': [2, 3, 4 ],
    'neurons': [ 32, 64],
    'activation': ['relu', 'tanh', 'swish'],
    #'learning_rate': [0.01, 0.001 ],
    #'batch_size': [ 32, 64 ],
    'optimizer': ['sgd', 'rmsprop', 'nadam', 'Adadelta'],
    'dropout_rate': [0.3 ],
    #'epochs': [ 8 ],
}

early_stopper = EarlyStopping(
    monitor='val_loss',    # Monitor validation loss
    min_delta=0.001,       # Minimum change to qualify as an improvement
    patience=10,           # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

results_df = pd.DataFrame(columns=['layers', 'neurons', 'activation', 'optimizer', 'dropout_rate', 'accuracy'])



mlflow.set_tracking_uri("file:./logs/")


def build_model(layers, neurons, activation, optimizer, dropout_rate ):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for _ in range(layers):
        model.add(tf.keras.layers.Conv2D(neurons, (3, 3), activation=activation, padding='same') )
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
    
        
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(  dropout_rate ) )
    
    model.add(tf.keras.layers.Dense(10, activation='softmax')) 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    #print("\n ___ Architecture " , i , "___ \n" )            
    print( layers, neurons, activation, optimizer, dropout_rate )
    #model.summary()
    
    
    return model


# Grid search
for layers in [2, 3, 4 ]:
    # If not the first iteration, update param_grid based on top 10 models
    if layers > 2:
        top_10_params = results_df[results_df['layers'] == layers-1].nlargest(10, 'accuracy')
        param_grid = {
            'neurons': top_10_params['neurons'].unique(),
            'activation': top_10_params['activation'].unique(),
            'optimizer': top_10_params['optimizer'].unique(),
            'dropout_rate': top_10_params['dropout_rate'].unique(),
        }

    for combination in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combination))
        params['layers'] = layers
    
        print( param_grid )
        	
        with mlflow.start_run(  ) as run :
                
            mlflow.log_params(params)
            
            model = build_model(layers=params['layers'], neurons=params['neurons'], activation=params['activation'], optimizer=params['optimizer'], dropout_rate=params['dropout_rate'] )
        
            i += 1
            model.fit(train_images, train_labels, epochs=100, batch_size=64, verbose=0, validation_split=0.1, callbacks=[early_stopper] )
            performance = model.evaluate(test_images, test_labels, verbose=1)[1]
            
            signature = infer_signature(train_images, model.predict(train_images))
            mlflow.set_tag("experiment_description", params )
                
            mlflow.log_metric("accuracy", performance)
            mlflow.tensorflow.log_model(model, "model"+str(i), signature=signature )
            mlflow.log_artifact("model.h5", "model")
            
            #_______________________________________________________________________________________        
            
            new_row = pd.DataFrame([{**params, 'accuracy': performance}])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            #_______________________________________________________________________________________            
            
            y_pred = model.predict(test_images)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(test_labels, axis=1)

            
            current_conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
            aggregate_conf_matrix += current_conf_matrix     

            plt.figure(figsize=(10,8))
            sns.heatmap(current_conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')

            conf_matrix_file = "confusion_matrix.png"
            plt.savefig(conf_matrix_file)
            plt.close()
            
            mlflow.log_artifact( conf_matrix_file )
            
            
            #_______________________________________________________________________________________
            
            last_dense_layer = model.layers[-1]
            intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=last_dense_layer.output)
            
            features = intermediate_layer_model.predict(test_images)

            # Run t-SNE on these features
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(features)

            # Plot t-SNE results
            plt.figure(figsize=(16,10))
            plt.scatter(tsne_results[:,0], tsne_results[:,1], c=np.argmax(test_labels, axis=1), cmap='tab10')
            plt.colorbar()

            # Save and log the plot
            tsne_plot_filename = f"tsne_visualization_arch_{i}.png"
            plt.savefig(tsne_plot_filename)
            plt.close()
            
            mlflow.log_artifact(tsne_plot_filename)
            


# Optional: save results_df to a CSV file for analysis
results_df.to_csv('grid_search_results.csv', index=False)

average_conf_matrix = aggregate_conf_matrix / i


df_average_conf_matrix = pd.DataFrame(average_conf_matrix, index=[f"True Class {i}" for i in range(num_classes)], columns=[f"Predicted Class {i}" for i in range(num_classes)])
csv_filename = "average_confusion_matrix.csv"
df_average_conf_matrix.to_csv(csv_filename)
mlflow.log_artifact(csv_filename)

# Convert the aggregate confusion matrix to a pandas DataFrame
df_aggregate_conf_matrix = pd.DataFrame(aggregate_conf_matrix, index=[f"True Class {i}" for i in range(num_classes)], columns=[f"Predicted Class {i}" for i in range(num_classes)])
aggregate_csv_filename = "aggregate_confusion_matrix.csv"
df_aggregate_conf_matrix.to_csv(aggregate_csv_filename)
mlflow.log_artifact(aggregate_csv_filename)

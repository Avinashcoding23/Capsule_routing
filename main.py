import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from capsule_layers import ConvCapsuleLayer, DenseCapsuleLayer, Length
from capsule_network import CapsuleNetwork
from scipy.stats import rankdata

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set the path for the data
train_path = 'C:/Users/Avinash/OneDrive/Desktop/Projects/Major Project/COVID-19/train'
test_path = 'C:/Users/Avinash/OneDrive/Desktop/Projects/Major Project/COVID-19/test'

# Define the image size and batch size
img_size = (224, 224)
batch_size = 16

# Create the ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and testing data using the ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Define the routing algorithm types to be used in the ensemble
routing_types = ['Dynamic', 'EM', 'FSA', 'None']

# Create an empty list to store the ensemble models
ensemble_models = []

# Train a model for each routing algorithm type
for routing_type in routing_types:
    # Define the capsule network architecture
    model = CapsuleNetwork(
        input_size=(224, 224, 3),
        n_class=1,
        routing_type=routing_type
    )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size,
        epochs=10
    )
    
    # Append the model to the ensemble list
    ensemble_models.append(model)

# Evaluate the performance of the ensemble on the test set
y_pred_ensemble = np.zeros_like(test_generator.classes)
ranks = []
for i, model in enumerate(ensemble_models):
    y_pred = model.predict(test_generator)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred_ensemble += y_pred
    ranks.append(rankdata(-np.array(test_generator.classes * 2 - 1) * np.array(y_pred * 2 - 1), method='min')[i])

# Calculate the weights based on the fuzzy rank
weights = np.array([rank / sum(ranks) for rank in ranks])

# Average the predictions using the weights
y_pred_ensemble = np.where(y_pred_ensemble >= (len(ensemble_models) / 2), 1, 0)
y_pred_final = np.average(np.array([model.predict(test_generator) for model in ensemble_models]), axis=0, weights=weights)
y_pred_final = np.where(y_pred_final > 0.5, 1, 0)

# Print the performance metrics
accuracy = np.mean(y_pred_final == test_generator.classes)
print("Accuracy: ", accuracy)

# Save each model in the ensemble_models list to a separate .h5 file
for i, model in enumerate(ensemble_models):
    model_name = f"model_{i}.h5"
    model.save(model_name)


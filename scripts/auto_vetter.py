import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pathlib

# Returns train_dataset, val_dataset - datasets for both training the model and validating it
def load_image_dataset(database_url, batch_size, img_height, img_width):
    archive = tf.keras.utils.get_file(origin=url, extract=True)

    # with_suffix('') removes file extension from path
    data_dir = pathlib.Path(archive).with_suffix('')

    # glob only goes into the immediate subdirectory of data_dir
    # First * matches and subdirectory within data_dir
    # / is a directory separator
    # Second * matches any file with .jpg extensions within the subdirectories
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f'Total images -> {image_count}')

    # List of all image paths of a class within a directory
    #roses = list(data_dir.glob('roses/*'))

    # Validation Split: 80% of images for training, 20% for validation
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_dataset.class_names
    train_dataset = configure_for_performance(train_dataset)
    val_dataset = configure_for_performance(val_dataset)

    # Must still be normalized in the model
    return train_dataset, val_dataset, class_names

# Optimizes the datasets by prefetching them
# Returns dataset from the cache
def configure_for_performance(dataset):
    # Uses Buffered Prefetching: To yield data from disk by loading data in advance even while processing current
    # without having the I/O (Input/Output) become blocking (Bottleneck/Stalling operations while waiting for tasks)

    # Dataset.cache: Keeps images in memory after being loaded off disk during first epoch
    # Dataset.fetch: Preprocesses next batch of data while model trains on current

    # AUTOTUNE automatically determines optimal buffer size for prefetching
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
# Retrieves batches of images and prints them
# dataset - dataset to visualize 
def retrieve_batch_images(dataset):
    # Tensor - Data structure (Multi-dimensional array)
    # Rank - Number of dimensions a tensor has 
    # Shape - Indicates how many elements are in each dimension of a tensor (tuple of ints)
    for image_batch, labels_batch in dataset:
        # image_batch - tensor of shape (# of images, R, G, B)
        print(image_batch.shape)

        # Corresponding labels to the # of images
        print(labels_batch.shape)
        break

# Defines a sequential model for images
# train_dataset - Dataset for training a model
# val_dataset - Dataset for validating a model
# class_names - Types of classes 
def create_image_model(class_names):
    num_classes = len(class_names)

    # Squential model - Linear stack of layers (Each output is input for next layer)
    # Layers - Function that transforms input to output
    model = tf.keras.Sequential([
        # Normalization Layer:
        # - Standardizes the images to between the ranges [0, 1] 
        tf.keras.layers.Rescaling(1./255),
        
        # First Covolution Block: Detects low-level features
        # - Conv2D: Convolutional layer with 32 filters, size 3x3
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(), # Reduces dimensions to abstract complex features
        
        # Second Covolution Block: Processes first block to detect mid-level features
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # Third Convolution Block: Processes high-level features to detect abtract parts
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        
        # Flatten Layer:
        # - Flattens 2D feature maps/activation maps: Outputs of convolution layers after applying their filters
        # - to 1D vectors (Multi-dimensional -> 1D array)
        # - Prepares for dense layers
        tf.keras.layers.Flatten(),

        # Dense Layer (128 neurons):
        # - Fully connected layer: Each neuron is connected to each other
        # - ReLU (Rectified Linear Unit) activation function: 
        #   - Defintion: ReLu(x) = max(0, x) (Positive unchanged, negative becomes 0) that is 
        #   - Non-linear: Able to learn complex patterns
        #   - Mitigates vanishing gradient problem: (gradient too small, slow learning) 
        #   - Mitigates exploding gradient problem: (gradient too large, unstable)
        tf.keras.layers.Dense(128, activation="relu"),

        # Dense Layer (num_classes neurons): Outputs logits for each class
        tf.keras.layers.Dense(num_classes)
    ])

    # Dropout Layer:
    # - During training, neurons have a (20%) chance of setting its input units to 0
    # - Regularization technique: Methods that reduce overfitting
    # - Prevents overfitting: (Model relies too much on training data and can't generalize on new data)
    # - During inference (Predictions): dropout isn't applied, but
    # - By inverted dropout: weights are scaled down by (1 - dropout rate) to account for it

    return model


# Defines a loss function: How well predictions match truth 
# Takes vector of ground (data from the source) truth values (Accurate/Reliable data) and vector of logits 
# Returns scalar loss for each: negative log probability (0 if sure of correct)
def define_loss_function():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compiles/configures a model
# Optimizier (adam): Adjusts weights and biases to minimize loss function 
#   - Gradient Descent: Update parameters in opposite direction to gradient of loss function
# Loss_function: How well predictions match truth
# Metrics: Evaluate performance (Doesn't influence training, but provides insight)
#   - Accuracy: fraction of correctly predicted samples
def compile_model(model, optimizer, loss_function, metrics):
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)


# Trains a model
# train_dataset - Inputs
# val_dataset - Ground truth labels
# epochs - How many times to iterate
def train_model(model, train_dataset, val_dataset, epochs):
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)


# Finds probability of a training subset
# model - neutal model 
# training_subset - Set to find probability of 
# Returns list of probability values 
def probabilities_for_training_subset(model, training_subset):
    # Calls model with training_subset as input
    # Model returns a vector of logits/log-odds (raw unnormalized output/log probabilities ln(p/(1-p)) scores, one for each class 
    # Converts to a numpy array (Easier to work with)
    predictions = model(training_subset).numpy()

    # Applies softmax layer from neural network module (nn): Activation function that converts logits into probabilities 
    # Possible to add to last layer of model, but this makes it impossible to provide loss calculations for models
    probabilities = tf.nn.softmax(predictions).numpy() 

    return probabilities


# Checks performance of a model
# val_dataset: Ground truth labels that correspond to input data
# verbose: How much info to show, (2) means minimal but essential info
def evaluate_results(model, test_dataset, class_names, verbose):
    loss, accuracy = model.evaluate(test_dataset, verbose=verbose)

    print(f'Loss -> {loss}')
    print(f'Accuracy -> {accuracy}')

    # Predicts logit for test_dataset
    logits = model.predict(test_dataset)

    # Axis=0 Operate along the rows
    # Axis=1 Operate along the columns 
    # Axis=2 Operate along the depth/third dimension
    # Argmax returns indices of max probability
    predicted_indices = tf.argmax(logits, axis=1).numpy()
    
    ground_truth_indices = []
    
    # Extends: Appends elements from an iterable (batch of labels) to ground_truth_classes 
    # Ignores inputs since we only need labels
    for input, labels in test_dataset:
        ground_truth_indices.extend(labels.numpy())
    
    # Converts list to tensor and then numpy array
    ground_truth_indices = tf.constant(ground_truth_indices).numpy()
    
    # Change list of indices to actual class names
    predicted_classes = [class_names[i] for i in predicted_indices]
    ground_truth_classes = [class_names[i] for i in ground_truth_indices]
        
    print(f'Predicted classes -> {predicted_classes}')
    print(f'Grounded truth values -> {ground_truth_classes}')


def main():
      # Number of samples
    database_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    train_dataset, val_dataset, class_names = load_image_dataset(database_url, batch_size = 32, img_height = 180, img_width = 180)
    
    create_image_model(class_names)

    # model = create_digit_model()
    #model = create_image_model(class_names)
    
    # Loads preexisting model
    model = tf.keras.models.load_model('vetting_model.keras')
    compile_model(model, optimizer = "adam", loss_function=define_loss_function(), metrics=['accuracy'])
    
    #train_model(model, train_dataset, val_dataset, epochs=3)

    # Grab first sample of the training set and print its probabilities
    # print(probabilities_for_training_subset(model, training_subset=train_dataset[:1]))
    evaluate_results(model, val_dataset, class_names, verbose=2)

    # Saves the vetting model into a keras file
    model.save('vetting_model.keras')

if __name__ == "__main__":
    pass#main()

import numpy as np
import tensorflow as tf
import cv2

def load_mnist():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test

def apply_gradient(image):
    # Apply gradient to the image
    grad_image = np.gradient(image.astype(float), axis=(0, 1))  # Apply gradient to the image
    gradient_magnitude = np.sqrt(grad_image[0]**2 + grad_image[1]**2)
    return gradient_magnitude

def reframe_image(image):
    # Find non-zero elements and crop out the black space
    coords = np.column_stack(np.where(image > 0))
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    reframed_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return reframed_image

def resize_image(image, size=(16, 16)):
    # Resize the image to fit N x N pixels
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def set_depth(image, m_bits):
    # Normalize to the desired bit-depth (grayscale)
    max_val = (2 ** m_bits) - 1
    return np.round(image / 255.0 * max_val).astype(np.uint16)  # Assuming original image is 0-255

def process_data(x_data, y_data, size=(16, 16), m_bits=1):
    processed_images = []
    labels = []
    
    for img, label in zip(x_data, y_data):
        img = apply_gradient(img)
        img = reframe_image(img)
        img = resize_image(img, size=size)
        img = set_depth(img, m_bits)
        processed_images.append(img)
        labels.append(label)
    
    return np.array(processed_images), np.array(labels)

def save_data(images, labels, filename='MNIST-simp.npz'):
    np.savez_compressed(filename, images=images, labels=labels)

def load_data(train_file='MNIST-simp-train.npz', test_file='MNIST-simp-test.npz'):
    # Load processed data
    train_data = np.load(train_file)
    test_data = np.load(test_file)

    x_train = train_data['images']
    y_train = train_data['labels']
    x_test = test_data['images']
    y_test = test_data['labels']

    # Normalize the data to 0-1 range for 4-bit depth (i.e., values between 0 and 15)
    x_train = x_train.astype('float32') / 15.0
    x_test = x_test.astype('float32') / 15.0

    # Reshape the data to have an additional dimension for channels (grayscale)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test

# Build the neural network
def build_model(input_shape=(16, 16, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),  # Flatten input image
        tf.keras.layers.Dense(16, activation='relu'),  # First hidden layer (16 units)
        tf.keras.layers.Dense(8, activation='relu'),   # Second hidden layer (8 units)
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer (10 units for 10 classes)
    ])
    
    # Add Dropout for regularization
    model.add(tf.keras.layers.Dropout(0.2))  # 20% dropout to avoid overfitting
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train and evaluate the model
def train_and_evaluate():
    # Load and process MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist()

    # Process training and test sets (using the simplifier functions)
    x_train_processed, y_train_processed = process_data(x_train, y_train, size=(16, 16), m_bits=4)
    x_test_processed, y_test_processed = process_data(x_test, y_test, size=(16, 16), m_bits=4)

    # Save processed data (simplified images)
    save_data(x_train_processed, y_train_processed, 'MNIST-simp-train.npz')
    save_data(x_test_processed, y_test_processed, 'MNIST-simp-test.npz')

    # Load the simplified data
    x_train_simplified, y_train_simplified, x_test_simplified, y_test_simplified = load_data()

    # Build the model
    model = build_model(input_shape=(16, 16, 1))

    # Train the model
    model.fit(x_train_simplified, y_train_simplified, epochs=7, batch_size=32, validation_data=(x_test_simplified, y_test_simplified), verbose=2)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(x_test_simplified, y_test_simplified)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Run the function to train and evaluate the model
train_and_evaluate()

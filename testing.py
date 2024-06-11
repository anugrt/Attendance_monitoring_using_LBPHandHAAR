import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_images_and_labels(dataset_path):
    face_images = []
    face_labels = []
    label_mapping = {}  # Dictionary to map person names to integer labels
    current_label = 0

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        if person_folder not in label_mapping:
            label_mapping[person_folder] = current_label
            current_label += 1

        label = label_mapping[person_folder]

        for file in os.listdir(person_path):
            image_path = os.path.join(person_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                face_images.append(image)
                face_labels.append(label)

    return face_images, face_labels

def train_lbph_model(train_images, train_labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_images, np.array(train_labels))
    return recognizer

def evaluate_recognizer(recognizer, test_images, test_labels):
    predicted_labels = []
    for image in test_images:
        label, _ = recognizer.predict(image)
        predicted_labels.append(label)

    accuracy = np.mean(np.array(predicted_labels) == np.array(test_labels))
    return accuracy

def test_accuracy(test_images, test_labels, recognizer):
    accuracies = []
    split_sizes = []

    # Vary the data split sizes from 10% to 90% for testing
    for split in range(1, 10):
        split_size = split / 10.0
        _, split_images, _, split_labels = train_test_split(
            test_images, test_labels, test_size=1 - split_size, random_state=42
        )

        accuracy = evaluate_recognizer(recognizer, split_images, split_labels)
        accuracies.append(accuracy)
        split_sizes.append(split_size)

    return split_sizes, accuracies

if __name__ == "__main__":
    # Replace 'path/to/training_dataset' with the path to your training dataset
    training_dataset_path = 'E:/projects/kmeaproject/Multiple-Face-Recognition-master (1)/training_data'
    face_images, face_labels = load_images_and_labels(training_dataset_path)

    # Split data into training and testing sets (80% training, 20% testing)
    train_images, test_images, train_labels, test_labels = train_test_split(
        face_images, face_labels, test_size=0.2, random_state=42
    )

    # Train LBPH model
    recognizer = train_lbph_model(train_images, train_labels)

    # Save the trained model to a file
    recognizer.write("lbph_model.yml")

    # Test accuracy and get results
    split_sizes, accuracies = test_accuracy(test_images, test_labels, recognizer)

    # Plot the accuracy graph
    plt.plot(split_sizes, accuracies, marker='o')
    plt.title('Accuracy vs. Testing Data Split Size')
    plt.xlabel('Testing Data Split Size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    # Print final accuracy for 80% training and 20% testing
    final_accuracy = evaluate_recognizer(recognizer, test_images, test_labels)
    print(f"Final Accuracy: {final_accuracy:.2f}")

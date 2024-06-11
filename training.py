import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
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

    accuracy = accuracy_score(test_labels, predicted_labels)
    return accuracy

def main():
    # Replace 'path/to/dataset' with the path to your dataset containing subfolders for each person
    dataset_path = 'E:/projects/kmeaproject/Multiple-Face-Recognition-master (1)/training_data'
    face_images, face_labels = load_images_and_labels(dataset_path)
    accuracies = []
    split_sizes = []
    # Split data into training and testing sets (80% training, 20% testing)
    face_images, face_labels = load_images_and_labels(dataset_path)
    print(face_labels)
    # Lists to store accuracy values and corresponding data split sizes
    accuracies = []
    split_sizes = []

    # Vary the data split sizes from 10% to 90% for training
    for split in range(1, 10):
        split_size = split / 10.0
        train_images, test_images, train_labels, test_labels = train_test_split(
            face_images, face_labels, test_size=1 - split_size, random_state=42
        )

        # Train LBPH model
        recognizer = train_lbph_model(train_images, train_labels)

        # Save the trained model to a file
        recognizer.write("lbph_model.yml")

        # Evaluate the model
        accuracy = evaluate_recognizer(recognizer, test_images, test_labels)
        print(f"Split Size: {split_size:.1f}, Accuracy: {accuracy:.2f}")

        # Store accuracy and split size for plotting
        accuracies.append(accuracy)
        split_sizes.append(split_size)

    # Plot the accuracy graph



if __name__ == "__main__":
    main()

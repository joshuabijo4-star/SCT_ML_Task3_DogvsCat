import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = r"C:\Users\Srethul\Desktop\Dogs vs Cats\train\train"

IMAGE_SIZE = 64
MAX_CATS = 1000
MAX_DOGS = 1000

print("="*60)
print("Dataset Path :", DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("\nERROR : Dataset path not found!")
    sys.exit()

files = os.listdir(DATASET_PATH)

print("Folder Found")
print("Total Files :", len(files))
print("First 10 Files :", files[:10])
print("="*60)

images = []
labels = []

cat_count = 0
dog_count = 0

print("\nLoading Images...")

for file in files:

    if not file.lower().endswith(".jpg"):
        continue

    
    if cat_count >= MAX_CATS and dog_count >= MAX_DOGS:
        break

    img_path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if file.startswith("cat"):

        if cat_count < MAX_CATS:
            images.append(img.flatten())
            labels.append(0)
            cat_count += 1

    elif file.startswith("dog"):

        if dog_count < MAX_DOGS:
            images.append(img.flatten())
            labels.append(1)
            dog_count += 1

images = np.array(images)
labels = np.array(labels)

print("\nDataset Loaded Successfully")
print("Cats Loaded :", cat_count)
print("Dogs Loaded :", dog_count)
print("Total Images :", len(images))

if len(images) == 0:
    print("\nNo images loaded.")
    sys.exit()


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTraining Images :", len(X_train))
print("Testing Images :", len(X_test))


# TRAIN SVM


print("\nTraining SVM Model...")

model = SVC(
    kernel="linear",
    C=1
)

model.fit(X_train, y_train)

print("Training Completed!")


# PREDICTIONS


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy :", round(accuracy*100,2), "%")

print("\nClassification Report\n")
print(classification_report(
    y_test,
    predictions,
    target_names=["Cat","Dog"]
))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))


# DISPLAY SAMPLE RESULTS


plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(
        X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE),
        cmap="gray"
    )

    actual = "Cat" if y_test[i]==0 else "Dog"
    predicted = "Cat" if predictions[i]==0 else "Dog"

    plt.title(f"Actual : {actual}\nPredicted : {predicted}")

    plt.axis("off")

plt.tight_layout()
plt.show()
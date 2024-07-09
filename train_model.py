import numpy as np
from pathlib import Path
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import skimage.io
from skimage.transform import resize
from PIL import Image
import os

def load_image_files(container_path, dimension=(104, 104)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "A Rice Disease detection dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)

    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    return Bunch(data=flat_data, target=target, target_names=categories, images=images, DESCR=descr)

# Load the dataset
container_path = "C:/project/4. RiceDiseasePrediction/datasets/"
image_dataset = load_image_files(container_path)
print("Loaded categories:", image_dataset.target_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

# Train the model
clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model
print("ACCURACY")
print(accuracy_score(y_test, y_pred) * 100)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# Save the model
model_filename = "rice_disease_model.pkl"
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")

# Save target names for later use
target_names_filename = "target_names.npy"
np.save(target_names_filename, image_dataset.target_names)
print(f"Target names saved to {target_names_filename}")

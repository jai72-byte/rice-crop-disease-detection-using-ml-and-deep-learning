import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import time


def load_image(file, dimension=(104, 104)):
    img = imread(file)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    flat_data = [img_resized.flatten()]
    return img, flat_data

# Load the model
model_filename = "rice_disease_model.pkl"
start_time = time.time()
clf = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load target names
target_names_filename = "target_names.npy"
target_names = np.load(target_names_filename)
print(f"Target names loaded from {target_names_filename}")

# Load and predict on a new image
image_path = r'C:\Users\91879\PycharmProjects\pythonProject5\datasets\Brown spot\DSC_0303.JPG'
plot_img, flat_img = load_image(image_path)

# Predict
prediction_start_time = time.time()
prediction = clf.predict(flat_img)
predicted_disease = target_names[prediction[0]]

# Measure and print times
end_time = time.time()
total_time = end_time - start_time
prediction_time = end_time - prediction_start_time
print(f"Total time: {total_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")

# Display the image and prediction
plt.imshow(plot_img)
plt.title(f"Predicted Disease: {predicted_disease}")
plt.show()

print("Predicted Disease is:", predicted_disease)

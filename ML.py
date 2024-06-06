import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

data_dir = 'D:\งานลูกค้า\Kp Kt\อ้อย'
categories = os.listdir(data_dir)
image_size = (64, 64)
print(categories)

def load_data(data_dir):
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = imread(img_path)
                img_resized = resize(img, image_size, anti_aliasing=True)
                images.append(img_resized.flatten())  # แปลงภาพเป็นเวกเตอร์
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

X, y = load_data(data_dir)
#train and test 80 - 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# บันทึกโมเดล
model_path = 'knn_model.joblib'
joblib.dump(knn, model_path)
print(f"Model saved to {model_path}")
import os 
os.system("clear")

from tensorflow.keras import layers, models
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import json


data_dir = "/mnt/c/Users/wdk33/OneDrive/Pulpit/concrete_dat"

batch_size = 32
img_size = (227, 227) 

X_train = tensorflow.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=2138, 
    image_size=img_size,
    batch_size=batch_size,
)

X_test = tensorflow.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=2138,
    image_size=img_size,
    batch_size=batch_size,
)
''''
#====================================================CNN
model_cnn = models.Sequential([
    layers.Input(shape=(227, 227, 3)),
    layers.Rescaling(1.0/255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

print("\n ========= CNN Architecture summary: =========\n")
model_cnn.summary()

print("Starting CNN training...")
history_cnn = model_cnn.fit(X_train, validation_data=X_test, epochs=5)
'''

#===============================================DNN
model_dnn = models.Sequential([
    layers.Input(shape=(227, 227, 3)), 
    layers.Rescaling(1.0/255),
    layers.Flatten(), 
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    
    layers.Dense(1, activation='sigmoid')
])

model_dnn.compile(optimizer='adam',
                           loss='binary_crossentropy', 
                           metrics=['accuracy'])

print("\n ========= DNN Architecture summary: =========\n")
model_dnn.summary()

print("Starting DNN training...")
history_dnn = model_dnn.fit(X_train, validation_data=X_test, epochs=5)

#loss
loss_trening_dnn = history_dnn.history['loss']
loss_test_dnn = history_dnn.history['val_loss']

loss_trening_cnn = history_cnn.history['loss']
loss_test_cnn = history_cnn.history['val_loss']

#accuracy
acc_trening_dnn = history_dnn.history['accuracy']
acc_test_dnn = history_dnn.history['val_accuracy']

acc_trening_cnn = history_cnn.history['accuracy']
acc_test_cnn = history_cnn.history['val_accuracy']

epochs_cnn = range(1, len(acc_trening_cnn) + 1)
epochs_dnn = range(1, len(acc_trening_dnn) + 1)

with open('history_cnn.json', 'w') as f:
    json.dump(history_cnn.history, f)

with open('history_dnn.json', 'w') as f:
    json.dump(history_dnn.history, f)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Loss Function Evolution', fontsize=16, fontweight='bold')


axes[0].plot(epochs_cnn, loss_trening_cnn, 'r', label='Training')
axes[0].plot(epochs_cnn, loss_test_cnn, 'b', linewidth=2, label='Testing')
axes[0].set_title('CNN')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_dnn, loss_trening_dnn, 'r', label='Training')
axes[1].plot(epochs_dnn, loss_test_dnn, 'b', linewidth=2, label='Testing')
axes[1].set_title('DNN')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Accuracy Function Evolution', fontsize=16, fontweight='bold')

axes[0].plot(epochs_cnn, acc_trening_cnn, 'r', label='Training')
axes[0].plot(epochs_cnn, acc_test_cnn, 'b', linewidth=2, label='Testing')
axes[0].set_title('CNN')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_dnn, acc_trening_dnn, 'r', label='Training')
axes[1].plot(epochs_dnn, acc_test_dnn, 'b', linewidth=2, label='Testing')
axes[1].set_title('DNN')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
plt.show()

'''
y_test = np.concatenate([labels for _, labels in X_test], axis=0)
pred_test = (model_cnn.predict(X_test) > 0.5).astype(int).flatten()

matrix = sklearn.metrics.confusion_matrix(y_test, pred_test)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=X_train.class_names)
disp.plot(cmap='Blues')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
'''

y_true = []
pred_test_probs_cnn = []
pred_test_probs_dnn = []

#preparation for matrix
for images, labels in X_test:
    y_true.extend(labels.numpy())
    pred_test_probs_cnn.extend(model_cnn.predict_on_batch(images))
    pred_test_probs_dnn.extend(model_dnn.predict_on_batch(images))

y_true = np.array(y_true)
pred_test_cnn = (np.array(pred_test_probs_cnn) > 0.5).astype(int).flatten()
pred_test_dnn = (np.array(pred_test_probs_dnn) > 0.5).astype(int).flatten()

#matrixes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')


matrix_cnn = sklearn.metrics.confusion_matrix(y_true, pred_test_cnn)
disp_cnn = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_cnn, display_labels=X_train.class_names)
disp_cnn.plot(cmap='Blues', ax=axes[0], colorbar=False) 
axes[0].set_title('CNN Confusion Matrix')

matrix_dnn = sklearn.metrics.confusion_matrix(y_true, pred_test_dnn)
disp_dnn = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix_dnn, display_labels=X_train.class_names)
disp_dnn.plot(cmap='Reds', ax=axes[1], colorbar=False) 
axes[1].set_title('DNN Confusion Matrix')

plt.tight_layout()
plt.savefig('confusion_matrices_compared.png', dpi=300)
plt.show()

from sklearn.metrics import accuracy_score

acc_cnn = accuracy_score(y_true, pred_test_cnn) * 100
acc_dnn = accuracy_score(y_true, pred_test_dnn) * 100

# Drukujemy wyniki w konsoli (formatujemy do 2 miejsc po przecinku)
print(f"\n\n\n\n")
print(f" CNN Accuracy Score: {acc_cnn:} %")
print(f" DNN Accuracy Score: {acc_dnn:} %")
print(f"\n\n\n\n")

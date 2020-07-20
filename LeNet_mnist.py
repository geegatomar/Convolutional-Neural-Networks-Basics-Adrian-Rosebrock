from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from LeNet import LeNet
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] accessing mnist...")
dataset = datasets.fetch_openml("mnist_784")
data = dataset.data

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

(X_train, X_test, y_train, y_test) = train_test_split(data/255.0, dataset.target.astype("int"), test_size=0.25)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=10)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
preds = model.predict(X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

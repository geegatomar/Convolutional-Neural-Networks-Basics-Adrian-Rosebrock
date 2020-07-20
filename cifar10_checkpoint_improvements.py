from sklearn.preprocessing import LabelBinarizer
from mini_vggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the weights directory")
args = vars(ap.parse_args())

print("[INFO] loading cifar10...")
((X_test, y_test), (X_train, y_train)) = cifar10.load_data()
X_train = X_train.astype("float")/255.0
X_test = X_test.astype("float")/255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])

checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] training network...")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64, callbacks=callbacks, verbose=2)




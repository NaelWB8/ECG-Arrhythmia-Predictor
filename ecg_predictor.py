import os, ast, random
import numpy as np
import pandas as pd
import wfdb
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# CONFIGURATION
PTBXL_PATH = os.path.join(os.path.dirname(__file__), "ptbxl")
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32
EPOCHS = 30
BASE_LR = 3e-4
TARGETS = ["Normal", "MI", "BBB"]

# DOWNLOAD PTB-XL DATASET
print("-" * 50)
print(f"Ensuring PTB-XL data is downloaded to: {PTBXL_PATH}")
os.makedirs(PTBXL_PATH, exist_ok=True)
wfdb.dl_database("ptb-xl", dl_dir=PTBXL_PATH)
print("✅ PTB-XL data ready.")
print("-" * 50)

# FOCAL LOSS
def focal_loss(gamma=1.5, alpha=None):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        fl = (1 - pt) ** gamma * ce
        if alpha is not None:
            a = tf.gather(alpha, y_true)
            fl = a * fl
        return tf.reduce_mean(fl)
    return loss

# INCEPTION BLOCK
def inception_block(x, filters=64, kernels=(3, 5, 7)):
    branches = []
    for k in kernels:
        b = layers.Conv1D(filters // len(kernels), k, padding="same", activation="relu")(x)
        branches.append(b)
    out = layers.Concatenate()(branches)
    out = layers.Conv1D(filters, 1, padding="same", activation="relu")(out)
    return out

# LOAD LABELS
Y = pd.read_csv(os.path.join(PTBXL_PATH, "ptbxl_database.csv"), index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
scp = pd.read_csv(os.path.join(PTBXL_PATH, "scp_statements.csv"), index_col=0)

bbb_codes = {"RBBB", "CRBBB", "IRBBB", "LBBB", "CLBBB"}

def map_target(scp_dict):
    keys = list(scp_dict.keys())
    classes = [scp.loc[k, "diagnostic_class"] if k in scp.index else None for k in keys]
    descs = [str(scp.loc[k, "description"]).lower() if k in scp.index else "" for k in keys]
    if any(c == "MI" for c in classes): return "MI"
    if any((c == "CD") and ("bundle branch block" in d or k in bbb_codes)
           for k, c, d in zip(keys, classes, descs)): return "BBB"
    if any(c == "NORM" for c in classes) or any("normal" in d for d in descs): return "Normal"
    return "Other"

Y["target"] = Y.scp_codes.apply(map_target)
Y = Y[Y["target"].isin(TARGETS)].copy()

# LOAD SIGNALS
signals, labels, patients = [], [], []
for idx, fname in Y.filename_lr.items():
    base = os.path.join(PTBXL_PATH, fname)
    if not os.path.exists(base + ".dat"):
        continue
    sig, _ = wfdb.rdsamp(base)
    signals.append(sig)
    labels.append(Y.loc[idx, "target"])
    patients.append(Y.loc[idx, "patient_id"])

X = np.stack(signals)
y_text = np.array(labels)
groups = np.array(patients)
print("Loaded:", X.shape, Counter(y_text))

# OVERSAMPLE BBB TO BALANCE CLASSES
bbb_idx = np.where(y_text == "BBB")[0]
extra = np.tile(bbb_idx, 2)
sel_idx = np.concatenate([np.arange(len(y_text)), extra])
np.random.shuffle(sel_idx)
X, y_text, groups = X[sel_idx], y_text[sel_idx], groups[sel_idx]
print("Balanced:", Counter(y_text))

# TRAIN/TEST SPLIT
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y_text, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train_text, y_test_text = y_text[train_idx], y_text[test_idx]

# NORMALIZATION
eps = 1e-6
X_train_mean = X_train.mean(axis=0, keepdims=True)
X_train_std = X_train.std(axis=0, keepdims=True)
X_train = (X_train - X_train_mean) / (X_train_std + eps)
X_test = (X_test - X_train_mean) / (X_train_std + eps)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# LABEL ENCODING
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test = le.transform(y_test_text)
num_classes = len(le.classes_)

# CLASS WEIGHTS FOR FOCAL LOSS
weights = {"Normal": 1.0, "MI": 1.0, "BBB": 2.0}
alpha = np.array([weights[c] for c in le.classes_], dtype=np.float32)
alpha_tensor = tf.constant(alpha, dtype=tf.float32)

# DATASETS
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4096).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# MODEL
inp = layers.Input(shape=(X.shape[1], X.shape[2]))
x = inception_block(inp, 64)
x = layers.MaxPooling1D(2)(x)
x = inception_block(x, 128)
x = layers.MaxPooling1D(2)(x)

x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
att = layers.Dense(1, activation="tanh")(x)
att = layers.Softmax(axis=1)(att)
x = layers.Multiply()([x, att])
x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inp, out)
opt = tf.keras.optimizers.AdamW(learning_rate=BASE_LR, weight_decay=1e-4)
model.compile(optimizer=opt, loss=focal_loss(gamma=1.5, alpha=alpha_tensor), metrics=["accuracy"])
model.summary()

# TRAINING
lr_sched = callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5)
early_stop = callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)

history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[lr_sched, early_stop, checkpoint], verbose=1)

# EVALUATION
probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

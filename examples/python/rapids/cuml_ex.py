import cuml
from cuml.datasets.classification import make_classification
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split
from cupy import asnumpy
from sklearn.metrics import accuracy_score

# cuml aims to mimic the scikit-learn API
# and functionality put perform the computation
# on GPUs

# setup synthetic data set
N_SAMPLES = 1000
N_FEATS = 10
N_CLASS = 2

# define RF parameters
N_EST = 25
MAX_DEPTH = 10

# generate synthetic data
X, y = make_classification(
    n_classes=N_CLASS, n_features=N_FEATS, n_samples=N_SAMPLES, random_state=44
)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44)

# create and train model
model = cuRF(max_depth=MAX_DEPTH, n_estimators=N_EST, random_state=44)
trained_model = model.fit(X_train, y_train)

# predict values
preds = model.predict(X_test)

cu_score = cuml.metrics.accuracy_score(y_test, preds)
sk_score = accuracy_score(asnumpy(y_test), asnumpy(preds))

print(f" cuml accuracy    : {cu_score:.4f}")
print(f" sklearn accuracy : {sk_score:.4f}")

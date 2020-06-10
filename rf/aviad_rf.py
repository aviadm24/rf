from sklearn.ensemble import RandomForestClassifier
import pandas as pd

file_path = "../csv/lt.csv"
test_path = "../csv/ltcurrent.csv"
df = pd.read_csv(file_path)
df_list = [x for _, x in df.groupby(["supplier_id", "PN"])]
# print(df_list[0].filter(items=['LateQty', 'LateLots', 'ConfirmedNo', 'datediffsupplier']))

df_list[0]['datediffsupplier'].replace(
    to_replace=['\\N'],
    value=0,
    inplace=True
)
train_x = df_list[0].filter(items=['LateQty', 'LateLots', 'ConfirmedNo', 'datediffsupplier'])
train_y = df_list[0].filter(items=['LTLate'])
# print(train_x.head())
model = RandomForestClassifier(n_estimators=4,
                               bootstrap=True,
                               max_features='sqrt')
# Fit on training data
model.fit(train_x, train_y.values.ravel())
# Actual class predictions
test_x = pd.read_csv(test_path, usecols=['LateQty', 'LateLots', 'ConfirmedNo', 'datediffsupplier'])
test_x['datediffsupplier'].replace(
    to_replace=['\\N'],
    value=0,
    inplace=True
)
print(test_x)
rf_predictions = model.predict(test_x)
print(rf_predictions)
# Probabilities for each class
rf_probs = model.predict_proba(test_x)[:, 1]
print(rf_probs)
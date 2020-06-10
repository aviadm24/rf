from sklearn.ensemble import RandomForestClassifier
import pandas as pd
RSEED = 50
df = pd.read_csv('../db/2011.csv').sample(100000, random_state=RSEED)
df.head()
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')
# Fit on training data
model.fit(train, train_labels)
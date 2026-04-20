import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n_samples = 2000

data = {
    'shine_score': np.random.randint(5, 100, n_samples),
    'redness_score': np.random.randint(0, 100, n_samples),
    'avg_brightness': np.random.randint(40, 250, n_samples)
}
df = pd.DataFrame(data)

def label_10_levels(row):
    r = row['redness_score']
    s = row['shine_score']
    
    if r > 45 or s < 5: return 9
    elif r > 25 or s < 10: return 8
    elif r > 8.0 or s < 20: return 7
    elif r > 6.0 or s < 30: return 6
    elif r > 7.0 or s < 40: return 5
    elif r > 5.0 or s < 50: return 4
    elif r > 4.0 or s < 60: return 3
    elif r > 3.0 or s < 70: return 2
    elif r > 1.0 or s < 80: return 1
    else: return 0

df['target'] = df.apply(label_10_levels, axis=1)

X = df[['shine_score', 'redness_score', 'avg_brightness']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'dog_health_10_level_brain.pkl')

print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
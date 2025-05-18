# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Simulate Dataset (Replace with real dataset for actual analysis)
np.random.seed(42)
n = 1000
data = {
    'Hour': np.random.randint(0, 24, n),
    'Day': np.random.randint(0, 7, n),
    'Visibility(mi)': np.random.uniform(0.5, 10.0, n),
    'Weather_Condition': np.random.randint(0, 5, n),  # Encoded: 0=Clear, 1=Rain, etc.
    'Severity': np.random.randint(1, 5, n)  # 1 (Low) to 4 (High)
}
df = pd.DataFrame(data)

# 3. Data Visualization
sns.countplot(x='Severity', data=df)
plt.title('Severity Distribution')
plt.show()

sns.boxplot(x='Severity', y='Visibility(mi)', data=df)
plt.title('Severity vs Visibility')
plt.show()

# 4. Train-Test Split
features = ['Hour', 'Day', 'Visibility(mi)', 'Weather_Condition']
X = df[features]
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))# Naan-mudhalvan-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Dataset load karo
print("Dataset load ho raha hai...")
df = pd.read_csv('dataset/hand_data.csv')

print(f"Total samples: {len(df)}")
print(f"Labels: {df['label'].value_counts().to_dict()}")

# X aur y alag karo
X = df.drop('label', axis=1).values
y = df['label'].values

# Train aur test split karo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model banao aur train karo
print("\nModel train ho raha hai...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test karo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Model save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model save ho gaya — model.pkl")
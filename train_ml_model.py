# train_ml_model.py - Train our AI!

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("RugGuard ML Training")
print("="*60)

# Load processed dataset
df = pd.read_csv('data/processed_dataset.csv')
print(f"Loaded {len(df)} contracts")

# Prepare data for ML
# X = features (everything except name, address, label)
# y = labels (SCAM=1, LEGIT=0)

feature_columns = [col for col in df.columns if col not in ['name', 'address', 'label']]
X = df[feature_columns]
y = df['label']

print(f"\nFeatures being used: {len(feature_columns)}")
print(feature_columns)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} contracts")
print(f"Test set: {len(X_test)} contracts")

# Train Random Forest (AI model)
print("\n🤖 Training AI model...")
model = RandomForestClassifier(
    n_estimators=100,  # 100 decision trees
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
print("✓ Training complete!")

# Test accuracy
print("\n📊 EVALUATING MODEL:")
print("="*60)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.1f}%")

# Detailed report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['LEGIT', 'SCAM']))

# Feature importance (what the AI learned)
print("\n🔍 MOST IMPORTANT FEATURES (What AI learned):")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Save model
joblib.dump(model, 'models/rugguard_model.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("\n✓ Model saved to models/rugguard_model.pkl")

print("\n" + "="*60)
if accuracy >= 0.85:
    print("🎉 EXCELLENT! Model is ready for deployment!")
elif accuracy >= 0.70:
    print("👍 GOOD! Model works. Can improve with more data.")
else:
    print("📚 Needs more training data. Add more contracts.")
print("="*60)
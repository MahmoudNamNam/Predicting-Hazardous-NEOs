import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

# Step 1: Data Importing and Cleaning
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Handle missing values
    df.dropna(inplace=True)
    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Feature selection
    features = df.drop(columns=['neo_id', 'name', 'orbiting_body', 'is_hazardous'])
    target = df['is_hazardous']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
    
    return X_train_res, X_test, y_train_res, y_test

# Step 3: Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"{model_name} - Mean CV ROC-AUC: {cv_scores.mean():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        report = classification_report(y_test, y_pred, target_names=['Not Hazardous', 'Hazardous'])
        roc_auc = roc_auc_score(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[model_name] = {
            'Classification Report': report,
            'ROC-AUC Score': roc_auc,
            'Confusion Matrix': conf_matrix
        }
        
        # Save the trained model
        model_path = f'models/{model_name.replace(" ", "_").lower()}_model.pkl'
        joblib.dump(model, model_path)
    
    return results

# Step 4: Model Saving and Reporting
def save_results_and_report(results, report_dir='reports'):
    os.makedirs(report_dir, exist_ok=True)
    for model_name, result in results.items():
        report_path = os.path.join(report_dir, f'{model_name.replace(" ", "_").lower()}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Classification Report:\n{result['Classification Report']}\n")
            f.write(f"ROC-AUC Score: {result['ROC-AUC Score']}\n")
            f.write(f"Confusion Matrix:\n{result['Confusion Matrix']}\n")
        print(f"Report saved for {model_name} at {report_path}")

if __name__ == "__main__":
    # Load and clean the data
    df = load_and_clean_data('./Data/nearest-earth-objects(1910-2024).csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save results and generate reports
    save_results_and_report(results)

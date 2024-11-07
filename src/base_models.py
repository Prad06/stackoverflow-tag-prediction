import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

def logisticRegressionOut(X, y, result_save_path="lr_results.txt", model_save_path="lr.joblib"):
    print("Starting Logistic Regression...")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print("Encoded")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Split Working")

    model = LogisticRegression(solver="saga", multi_class="multinomial", penalty="l2", max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    with open(result_save_path, 'w') as f:
        f.write("Logistic Regression Results\n")
        f.write("===========================\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    print(f"Logistic Regression results saved to {result_save_path}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.savefig("lr_confusion_matrix.png")
    plt.close()

    if model_save_path:
        joblib.dump(
            {"model": model, "label_encoder": label_encoder}, model_save_path
        )
        print(f"Model saved to {model_save_path}")

    print("OUT FROM LR")


def randomForestOut(X, y, save_path="rf_results.txt", model_save_path= "randomForest.joblib"):
    print("Starting Random Forest...")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    tree_sizes = [400, 500, 600]
    test_errors = []
    train_errors = []
    f1_scores = []
    auc_scores = []
    imp_features = []
    
    kf = KFold(n_splits=10, shuffle=True)
    
    for tree_size in tree_sizes:
        print(f"Training model with {tree_size} trees...")
        model = RandomForestClassifier(
            n_estimators=tree_size, 
            n_jobs=-1, 
            random_state=42, 
            bootstrap=True, 
            oob_score=True, 
            max_depth=30, 
            max_features=0.01,
            class_weight='balanced'
        )
        
        kf_errors = []
        for train_ind, val_ind in kf.split(X_train):
            if scipy.sparse.issparse(X_train):
                X_train_kf = X_train[train_ind]
                X_val = X_train[val_ind]
            else:
                X_train_kf = X_train.iloc[train_ind]
                X_val = X_train.iloc[val_ind]
            y_train_kf = y_train[train_ind]
            y_val = y_train[val_ind]
            
            model.fit(X_train_kf, y_train_kf)
            y_pred_val = model.predict(X_val)
            kf_error = 1 - accuracy_score(y_val, y_pred_val)
            kf_errors.append(kf_error)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        y_pred_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        test_errors.append(test_error)
        train_errors.append(np.mean(kf_errors))
        f1_scores.append(f1)
        auc_scores.append(auc_score)
        
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        imp_features.append(indices[:10])
        
        print(f"Trees: {tree_size}, Test Error: {test_error:.4f}, Train Error: {np.mean(kf_errors):.4f}, F1 Score: {f1:.4f}, AUC Score: {auc_score:.4f}")
    
    with open(save_path, 'w') as f:
        f.write("Random Forest Results\n")
        f.write("=====================\n\n")
        f.write("Performance Metrics:\n")
        for i, tree_size in enumerate(tree_sizes):
            f.write(f"Trees: {tree_size}\n")
            f.write(f"Test Error: {test_errors[i]:.4f}\n")
            f.write(f"Train Error: {train_errors[i]:.4f}\n")
            f.write(f"F1 Score: {f1_scores[i]:.4f}\n")
            f.write(f"AUC Score: {auc_scores[i]:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print(f"Random Forest results saved to {save_path}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Random Forest - Confusion Matrix')
    plt.savefig("rf_confusion_matrix.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(tree_sizes, test_errors, label="Test error")
    plt.plot(tree_sizes, train_errors, label="Train error")
    plt.plot(tree_sizes, f1_scores, label="F1 score")
    plt.plot(tree_sizes, auc_scores, label="ROC AUC")
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.title("Random Forest - Model Performance vs Number of Trees")
    plt.legend()
    plt.savefig("rf_performance_plot.png")
    plt.close()

    if model_save_path:
        joblib.dump(
            {"model": model, "label_encoder": label_encoder}, model_save_path
        )
        print(f"Model saved to {model_save_path}")

    print("OUT FROM RF")

def svmOut(X, y, save_path="svm_results.txt", model_save_path="svm_model.joblib"):
    print("Starting SVM...")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True)
    kf_errors = []
    
    for train_ind, val_ind in kf.split(X_train):
        if scipy.sparse.issparse(X_train):
            X_train_kf = X_train[train_ind]
            X_val = X_train[val_ind]
        else:
            X_train_kf = X_train.iloc[train_ind]
            X_val = X_train.iloc[val_ind]
        y_train_kf = y_train[train_ind]
        y_val = y_train[val_ind]
        
        model.fit(X_train_kf, y_train_kf)
        y_pred_val = model.predict(X_val)
        kf_error = 1 - accuracy_score(y_val, y_pred_val)
        kf_errors.append(kf_error)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    test_error = 1 - accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    y_pred_proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    with open(save_path, 'w') as f:
        f.write("SVM Results\n")
        f.write("===========\n\n")
        f.write(f"Test Error: {test_error:.4f}\n")
        f.write(f"Cross-validation Error: {np.mean(kf_errors):.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC Score: {auc_score:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print(f"SVM results saved to {save_path}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SVM - Confusion Matrix")
    plt.savefig("svm_confusion_matrix.png")
    plt.close()
    
    if model_save_path:
        joblib.dump(
            {"model": model, "label_encoder": label_encoder}, model_save_path
        )
        print(f"Model saved to {model_save_path}")

    print("OUT FROM SVM")
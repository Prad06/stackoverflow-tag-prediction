import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def distilbertOut(
    df,
    save_path="../results/distilbert_results.txt",
    model_save_path="../models/distilbert_model.joblib",
    epochs=10,
    batch_size=16,
    learning_rate=1e-5,
    test_size=0.2,
):
    print("Starting DistilBERT...")

    label_encoder = LabelEncoder()
    df["encoded_label"] = label_encoder.fit_transform(df["Label"])

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    def reset_ready(df):
        df.reset_index(inplace=True)
        df = df[["Title", "encoded_label"]]
        return df

    # Create DataFrames
    test_df = reset_ready(test_df)
    train_df = reset_ready(train_df)
    val_df = reset_ready(val_df)

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["Title"], padding=True, truncation=True)

    dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)
    dataset_encoded = dataset_encoded.rename_column("encoded_label", "labels")

    # Model
    model_ckpt = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=5
    ).to(device)

    # Training arguments
    logging_steps = len(dataset_encoded["train"]) // batch_size
    training_args = TrainingArguments(
        output_dir=f"{model_ckpt}-finetuned-stackoverflow",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        log_level="error",
        report_to="none",
    )

    # Metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["validation"],
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate(dataset_encoded["test"])

    # Predictions
    preds_output = trainer.predict(dataset_encoded["test"])
    y_preds = np.argmax(preds_output.predictions, axis=1)

    # Save results
    with open(save_path, "w") as f:
        f.write("DistilBERT Results\n")
        f.write("===================\n\n")
        f.write(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}\n")
        f.write(f"Test F1 Score: {eval_results['eval_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{compute_metrics(preds_output)}\n")

    print(f"DistilBERT results saved to {save_path}")

    # Plot confusion matrix
    true_labels = np.array(dataset_encoded["test"]["labels"])
    print(true_labels)
    print()
    print(y_preds)
    print()
    print(len(true_labels), len(y_preds))

    cm = confusion_matrix(true_labels, y_preds, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("../results/distilbert_confusion_matrix.png")
    plt.close()

    # Save model
    if model_save_path:
        joblib.dump(
            {"model": model, "tokenizer": tokenizer, "label_encoder": label_encoder},
            model_save_path,
        )
        print(f"\nModel saved to {model_save_path}")

    return model, tokenizer, label_encoder

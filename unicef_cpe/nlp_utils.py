import json
import os
from pathlib import Path


class Translator:
    """
    A reusable translator class that uses Lingua for language detection
    and M2M100 for translating text to English.
    """

    def __init__(self, translation_model="facebook/m2m100_418M"):
        from lingua import Language, LanguageDetectorBuilder
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        # Initialize Lingua language detector
        self.detector = LanguageDetectorBuilder.from_all_languages().build()

        # Load the translation model
        self.translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model)
        self.translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model)
        self.target_lang = "en"

    def detect_language(self, text):
        """
        Detects the language of the input text using Lingua.

        Args:
            text (str): The input text.

        Returns:
            str: The ISO 639-1 language code of the detected language.
        """
        detected_language = self.detector.detect_language_of(text)
        if detected_language is None:
            raise ValueError("Unable to detect language")
        return detected_language.iso_code_639_1.name.lower()

    def translate(self, text):
        """
        Translates the input text to English.

        Args:
            text (str): The input text in any supported language.

        Returns:
            str: The translated text in English.
        """
        # Detect the source language
        detected_lang = self.detect_language(text)

        if detected_lang == "en":
            return text

        self.translation_tokenizer.src_lang = detected_lang

        # Tokenize the input text
        inputs = self.translation_tokenizer(text, return_tensors="pt")

        # Generate the translation
        translated_tokens = self.translation_model.generate(
            **inputs,
            forced_bos_token_id=self.translation_tokenizer.get_lang_id(self.target_lang)
        )

        # Decode the translated tokens to a string
        translation = self.translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translation
    

def train_multilabel_classifier(df, model_name = 'distilbert-base-multilingual-cased'):
    """
    Trains a multi-class multi-label classifier on the given DataFrame.
    Returns the trained model, tokenizer, and id2label mapping.
    """
    # Imports
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
    from datasets import Dataset
    from itertools import chain
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Get all unique labels (goal areas)
    all_labels = set(chain.from_iterable(df['labels']))
    labels_list = sorted(list(all_labels))
    num_labels = len(labels_list)

    # Create label mappings
    label2id = {label: idx for idx, label in enumerate(labels_list)}
    id2label = {idx: label for idx, label in enumerate(labels_list)}

    # Convert DataFrame to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Encode labels into multi-hot vectors with float values
    def encode_labels(example):
        labels = example['labels']
        label_ids = [label2id[label] for label in labels]
        multi_hot = [0.0] * num_labels
        for idx in label_ids:
            multi_hot[idx] = 1.0
        example['labels'] = multi_hot
        return example

    dataset = dataset.map(encode_labels)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize text data
    def tokenize_function(examples):
        return tokenizer(
            examples['processed_text'],
            truncation=True,
            padding=False,
            max_length=256,  # Adjust max_length as needed
        )

    dataset = dataset.map(tokenize_function, batched=True)

    # Ensure labels are of type float32
    def cast_labels_to_float(batch):
        batch['labels'] = [list(map(float, labels)) for labels in batch['labels']]
        return batch

    dataset = dataset.map(cast_labels_to_float, batched=True)

    # Set format for PyTorch
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Split dataset into training and evaluation sets
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type='multi_label_classification',
        id2label=id2label,
        label2id=label2id,
    )

    # Define training arguments with use_cpu=True
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Adjust as needed
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        use_cpu=True,  # Force CPU training
    )

    # Define metrics for evaluation
    def compute_metrics(p):
        logits = torch.tensor(p.predictions)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        labels = torch.tensor(p.label_ids)

        f1 = f1_score(labels, preds, average='micro', zero_division=0)
        precision = precision_score(labels, preds, average='micro', zero_division=0)
        recall = recall_score(labels, preds, average='micro', zero_division=0)
        accuracy = accuracy_score(labels, preds)

        return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

    # Create data collator (no changes needed)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    return model, tokenizer, label2id, id2label


def predict_goal_areas(texts, model, tokenizer, id2label):
    """
    Predicts goal areas for a list of texts using the trained multi-label classifier.

    Args:
        texts (list of str): The input texts for which to predict goal areas.
        model (PreTrainedModel): The trained HuggingFace model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        id2label (dict): A mapping from label IDs to label names.

    Returns:
        list of list of str: A list where each element is a list of predicted goal areas for the corresponding text.
    """
    import torch

    # Ensure the model is in evaluation mode
    model.eval()

    # Set the device to CPU (since training was done with use_cpu=True)
    device = torch.device('cpu')
    model.to(device)

    # Tokenize the input texts
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding=True,  # Pad to the longest sequence in the batch
        max_length=256,  # Use the same max_length as during training
    )

    # Move inputs to the CPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and apply sigmoid to get probabilities
    logits = outputs.logits
    probs = torch.sigmoid(logits)

    # Convert probabilities to binary predictions (threshold at 0.5)
    preds = (probs > 0.5).int()

    # Map predictions to label names
    result = []
    for pred in preds:
        # Get indices of predicted labels
        pred_indices = pred.nonzero().flatten().tolist()
        # Ensure id2label keys are integers
        id2label = {int(k): v for k, v in id2label.items()}
        # Map indices to label names
        pred_labels = [id2label[idx] for idx in pred_indices]
        result.append(pred_labels)

    return result

    

def save_model_and_tokenizer(model, tokenizer, label2id, id2label, save_directory):
    """
    Saves the model, tokenizer, and label mappings to the specified directory.
    """
    

    # Ensure save directory exists
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Save the model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Save label mappings
    label_mappings = {
        "id2label": id2label,
        "label2id": label2id
    }
    with open(f"{save_directory}/label_mappings.json", "w") as fp:
        json.dump(label_mappings, fp)


def load_model_and_tokenizer(save_directory):
    """
    Loads the model, tokenizer, and label mappings from the specified directory.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)

    # Load label mappings from JSON
    label_mappings_path = Path(save_directory) / "label_mappings.json"
    with open(label_mappings_path, "r") as fp:
        label_mappings = json.load(fp)
        # Convert id2label keys to integers
        id2label = {int(k): v for k, v in label_mappings["id2label"].items()}
        label2id = label_mappings["label2id"]

    return model, tokenizer, id2label, label2id

# Yelp Review Sentiment Analysis with DistilBERT and LoRA
This repository contains the code and resources for fine-tuning a DistilBERT model on the Yelp review dataset to perform sentiment analysis. The fine-tuning process was enhanced using the LoRA (Low-Rank Adaptation) technique, leading to significant improvements in model accuracy and efficiency.

# Project Overview
This project demonstrates how to effectively fine-tune a pre-trained transformer model for a specific task—in this case, sentiment classification on Yelp reviews. By utilizing the Yelp Polarity dataset, we aimed to classify reviews as either positive or negative. The model was initially a base DistilBERT model, which was then fine-tuned to achieve a substantial performance boost.

# Key Achievements
Model: DistilBERT (pre-trained on uncased English text)
Dataset: Yelp Polarity Dataset
Technique: Fine-tuning with LoRA (Low-Rank Adaptation)
Performance Improvement: Increased classification accuracy from 50% to 92%, representing a 40% improvement.

# How to Use
Installation
To run this project, you need to install the required Python packages. You can install them using pip:
```
pip install transformers datasets peft
```
# Running the Code
Load the Dataset: The Yelp Polarity dataset is loaded using the datasets library.

```
from datasets import load_dataset
dataset = load_dataset("yelp_polarity")
```
# Model Setup: The DistilBERT model is configured for sequence classification.

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```
# Fine-Tuning with LoRA: The LoRA technique is applied to enhance the model's adaptability to the Yelp dataset.

```
from peft import get_peft_model, LoraConfig, TaskType

# Configure LoRA
config = LoraConfig(task_type=TaskType.SEQUENCE_CLASSIFICATION)
lora_model = get_peft_model(model, config)
```
# Training the Model: The model is trained using the Hugging Face Trainer class.

```
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt_lora_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```
# Saving the Model: After fine-tuning, the model is saved for future use.
```
lora_model.save_pretrained("./gpt_lora_output")
```

# Evaluation
After training, the model achieved a remarkable accuracy improvement on the test set, significantly outperforming its initial state.

## Results
* Initial Accuracy: 50%
* Final Accuracy: 92%
* Improvement: 40% increase in accuracy
  
# Conclusion
This project illustrates the power of fine-tuning pre-trained models with advanced techniques like LoRA to achieve high performance on specific tasks. The substantial accuracy improvement on the Yelp review classification task demonstrates the effectiveness of this approach.

# Future Work
Potential future improvements could include:

* Hyperparameter Optimization: Experimenting with different learning rates, batch sizes, and epochs.
* Model Exploration: Testing other transformer models for better baseline performance.
* Deployment: Creating a web interface to deploy the model for real-time sentiment analysis.

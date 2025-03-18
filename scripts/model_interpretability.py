import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
from transformers import pipeline

# Function to load the NER model
def load_ner_model(model_name):
    return pipeline("ner", model=model_name, tokenizer=model_name)

# Function to explain predictions using SHAP
def explain_with_shap(model, texts):
    # Tokenize inputs
    inputs = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    # Get model predictions
    outputs = model.model(**inputs).logits
    predictions = outputs.argmax(dim=-1).detach().cpu().numpy()
    
    # Create SHAP explainer
    explainer = shap.Explainer(model.model)
    shap_values = explainer(inputs["input_ids"])
    
    return shap_values, predictions

# Function to explain predictions using LIME
def explain_with_lime(model, texts):
    explainer = LimeTextExplainer(class_names=model.model.config.id2label.values())
    
    # Define prediction function for LIME
    def predict_fn(texts):
        inputs = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.model(**inputs).logits
        return outputs.softmax(dim=-1).detach().cpu().numpy()
    
    # Generate explanations
    explanations = []
    for text in texts:
        exp = explainer.explain_instance(text, predict_fn, num_features=10)
        explanations.append(exp)
    
    return explanations

# Function to analyze difficult cases
def analyze_difficult_cases(model, difficult_texts):
    shap_values, predictions = explain_with_shap(model, difficult_texts)
    lime_explanations = explain_with_lime(model, difficult_texts)
    
    return shap_values, predictions, lime_explanations

# Function to visualize SHAP values
def visualize_shap(shap_values):
    shap.summary_plot(shap_values)

# Function to visualize LIME explanations
def visualize_lime(explanations):
    for exp in explanations:
        exp.as_pyplot_figure()

# Main function to run interpretability analysis
def interpret_model(model_name, texts, difficult_texts):
    model = load_ner_model(model_name)
    
    # Explain predictions on general texts
    shap_values, predictions = explain_with_shap(model, texts)
    lime_explanations = explain_with_lime(model, texts)
    
    # Analyze difficult cases
    shap_values_difficult, predictions_difficult, lime_explanations_difficult = analyze_difficult_cases(model, difficult_texts)
    
    # Visualize results
    visualize_shap(shap_values_difficult)
    visualize_lime(lime_explanations_difficult)

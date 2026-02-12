

import torch
from transformer_lens import HookedTransformer
import random

def create_ioi_prompt(name1, name2, place, obj, duplicate_first=True):
    """
    Returns:
        dict with prompt, correct_answer, incorrect_answer, and metadata
    """

    if duplicate_first:
        duplicated = name1
        correct_answer = name2
    else:
        duplicated = name2
        correct_answer = name1
    
    prompt = f"When {name1} and {name2} went to the {place}, {duplicated} gave a {obj} to"
    
    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "incorrect_answer": duplicated,
        "name1": name1,
        "name2": name2,
        "place": place,
        "object": obj
    }

def compute_logit_difference(model, prompt_text, correct_name, incorrect_name):
    """
    Returns:
        tuple: (logit_diff, correct_logit, incorrect_logit)
    """

    # Tokenize and run model
    tokens = model.to_tokens(prompt_text)
    logits = model(tokens)
    
    # Get logits at final position
    final_logits = logits[0, -1, :]
    
    # Get token IDs for the two names (with leading space)
    correct_token_id = model.to_single_token(" " + correct_name)
    incorrect_token_id = model.to_single_token(" " + incorrect_name)
    
    # Extract logits
    correct_logit = final_logits[correct_token_id].item()
    incorrect_logit = final_logits[incorrect_token_id].item()
    
    logit_diff = correct_logit - incorrect_logit
    
    return logit_diff, correct_logit, incorrect_logit

# Verified single-token names in GPT-2
SINGLE_TOKEN_NAMES = [
    "John", "Mary", "Michael", "Sarah", "David", "Emma",
    "James", "Lisa", "Robert", "Anna", "William", "Emily",
    "Thomas", "Alice", "Daniel", "Sophie", "Matthew", "Grace",
    "Joseph", "Hannah", "Christopher", "Olivia", "Andrew", "Lucy"
    "Joshua", "Sophia", "Ryan", "Charlotte", "Nicholas", "Amelia"
]

PLACES = ["store", "park", "school", "office", "house"]
OBJECTS = ["drink", "gift", "book", "letter", "message"]

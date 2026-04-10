import json
import random
from typing import Optional


SINGLE_TOKEN_NAMES = [
    "John", "Mary", "Michael", "Sarah", "David", "Emma",
    "James", "Lisa", "Robert", "Anna", "William", "Emily",
    "Thomas", "Alice", "Daniel", "Sophie", "Matthew", "Grace",
    "Joseph", "Hannah", "Christopher", "Olivia", "Andrew", "Lucy",
    "Joshua", "Sophia", "Ryan", "Charlotte", "Nicholas", "Amelia"
]


TEMPLATES = {
    "template_1": {
        "structure": "When {name1} and {name2} went to the {place}, {duplicated} gave a {obj} to",
        "weight": 0.60,
        "description": "Original - When connector, went/gave verbs"
    },
    "template_2": {
        "structure": "While {name1} and {name2} were working at the {place}, {duplicated} gave a {obj} to",
        "weight": 0.25,
        "description": "Progressive tense - While connector, working/gave verbs"
    },
    "template_3": {
        "structure": "Friends {name1} and {name2} found a {obj} at the {place}. {duplicated} gave it to",
        "weight": 0.15,
        "description": "Different structure - Friends prefix, found/gave verbs"
    }
}

PLACES = ["store", "park", "school", "office", "house"]
OBJECTS = ["drink", "gift", "book", "letter", "message"]


# Prompt construction
def create_ioi_prompt(
    name1: str,
    name2: str,
    place: str,
    obj: str,
    duplicate_first: bool = True,
    template_name: str = "template_1"
) -> dict:
    """
    Construct a single IOI prompt.

    In BABA structure (duplicate_first=True): name1 appears twice → answer is name2.
    In ABBA structure (duplicate_first=False): name2 appears twice → answer is name1.

    """
    duplicated = name1 if duplicate_first else name2
    correct_answer = name2 if duplicate_first else name1

    structure = TEMPLATES[template_name]["structure"]
    prompt = structure.format(
        name1=name1,
        name2=name2,
        place=place,
        obj=obj,
        duplicated=duplicated
    )

    return {
        "prompt": prompt,
        "correct_answer": correct_answer,
        "incorrect_answer": duplicated,
        "name1": name1,
        "name2": name2,
        "place": place,
        "object": obj,
        "template": template_name,
        "duplicate_first": duplicate_first
    }


def create_corrupted_prompt(prompt_data: dict) -> dict:
    """
    Create the corrupted counterpart of a clean IOI prompt.

    Swaps which name is duplicated, so the correct answer changes.
    Used for activation patching: corrupted activations carry the
    wrong-answer signal and can be patched into clean runs.
    """
    corrupted = prompt_data.copy()
    corrupted["duplicate_first"] = not prompt_data["duplicate_first"]

    duplicated = corrupted["name1"] if corrupted["duplicate_first"] else corrupted["name2"]
    correct_answer = corrupted["name2"] if corrupted["duplicate_first"] else corrupted["name1"]

    structure = TEMPLATES[prompt_data["template"]]["structure"]
    corrupted["prompt"] = structure.format(
        name1=corrupted["name1"],
        name2=corrupted["name2"],
        place=corrupted["place"],
        obj=corrupted["object"],
        duplicated=duplicated
    )
    corrupted["correct_answer"] = correct_answer
    corrupted["incorrect_answer"] = duplicated

    return corrupted



def generate_ioi_dataset(
    n_total: int = 1000,
    names: Optional[list] = None,
    seed: int = 42
) -> list:

    if names is None:
        names = SINGLE_TOKEN_NAMES

    random.seed(seed)

    prompts_per_template = {
        name: int(n_total * tmpl["weight"])
        for name, tmpl in TEMPLATES.items()
    }

    dataset = []
    for template_name, n_prompts in prompts_per_template.items():
        for _ in range(n_prompts):
            name1, name2 = random.sample(names, 2)
            place = random.choice(PLACES)
            obj = random.choice(OBJECTS)
            duplicate_first = random.choice([True, False])

            dataset.append(
                create_ioi_prompt(name1, name2, place, obj, duplicate_first, template_name)
            )

    return dataset


def save_dataset(dataset: list, path: str) -> None:
    """Save dataset to JSON."""
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} prompts to {path}")


def load_dataset(path: str) -> list:
    """Load dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def generate_abc_dataset(
    n_total: int = 1000,
    names: Optional[list] = None,
    seed: int = 123
) -> list:
    """
    Generate a pABC reference dataset for mean ablation.

    Each prompt uses three distinct names (A, B, C) with no repetition,
    preserving the grammatical structure of IOI templates but removing
    the duplicate token signal. 
    """
    if names is None:
        names = SINGLE_TOKEN_NAMES

    random.seed(seed)

    prompts_per_template = {
        name: int(n_total * tmpl["weight"])
        for name, tmpl in TEMPLATES.items()
    }

    dataset = []
    for template_name, n_prompts in prompts_per_template.items():
        for _ in range(n_prompts):
            name_a, name_b, name_c = random.sample(names, 3)
            place = random.choice(PLACES)
            obj   = random.choice(OBJECTS)

            prompt = TEMPLATES[template_name]["structure"].format(
                name1=name_a, name2=name_b,
                place=place, obj=obj,
                duplicated=name_c
            )
            dataset.append({
                "prompt":   prompt,
                "name_a":   name_a,
                "name_b":   name_b,
                "name_c":   name_c,
                "template": template_name,
                "place":    place,
                "object":   obj
            })

    return dataset

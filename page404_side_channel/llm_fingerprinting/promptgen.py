"""
promptgen.py
Reusable random prompt generation library
Uses OS-level entropy via secrets for maximum randomness
"""

import secrets


SUBJECTS = [
    "neural networks", "machine learning", "quantum computing", "blockchain",
    "climate change", "natural selection", "encryption", "operating systems",
    "computer networks", "supply and demand", "photosynthesis", "vaccines",
    "black holes", "solar system", "human immune system", "DNA",
    "recursion", "memory management", "compiler design",
    "distributed systems", "transformers", "cryptography",
    "cloud computing", "database indexing", "hash tables",
    "sorting algorithms", "deep learning", "cybersecurity",
    "data privacy", "economics", "game theory",
    "plate tectonics", "water cycle", "thermodynamics"
]

VERBS = [
    "Explain", "Describe", "Summarize", "Analyze", "Compare",
    "Discuss", "Outline", "Illustrate", "Evaluate", "Examine"
]

MODIFIERS = [
    "in simple terms",
    "for a beginner",
    "step by step",
    "using an analogy",
    "with real world examples",
    "in technical detail",
    "in a short paragraph",
    "like I'm five years old"
]

COMPARISONS = [
    "TCP and UDP",
    "SQL and NoSQL",
    "Python and C++",
    "public key and symmetric encryption",
    "supervised and unsupervised learning",
    "monolith and microservices"
]

TEMPLATES = [
    "{verb} how {subject} works {modifier}.",
    "{verb} the key principles behind {subject} {modifier}.",
    "{verb} why {subject} is important.",
    "{verb} the real-world applications of {subject}.",
    "Provide a short explanation of {subject} {modifier}.",
    "Create a beginner-friendly guide to {subject}.",
    "Compare and contrast {comparison}.",
    "Explain the difference between {comparison}.",
    "List practical applications of {subject}.",
]


def generate_prompt():
    """
    Generate a random prompt using OS entropy.
    """
    template = secrets.choice(TEMPLATES)

    return template.format(
        verb=secrets.choice(VERBS),
        subject=secrets.choice(SUBJECTS),
        modifier=secrets.choice(MODIFIERS),
        comparison=secrets.choice(COMPARISONS)
    )


_USED = set()

def generate_unique_prompt():
    """
    Generate a prompt that has not been returned before
    during this runtime session.
    """
    while True:
        p = generate_prompt()
        if p not in _USED:
            _USED.add(p)
            return p


def generate_batch(n=10, unique=False):
    """
    Generate a batch of prompts.

    Args:
        n (int): number of prompts
        unique (bool): ensure uniqueness in this batch

    Returns:
        list[str]
    """
    if unique:
        return [generate_unique_prompt() for _ in range(n)]
    else:
        return [generate_prompt() for _ in range(n)]

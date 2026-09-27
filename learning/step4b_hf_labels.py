import torch
from transformers import pipeline

device = "mps" if torch.backends.mps.is_available() else "cpu"
tagger = pipeline("zero-shot-classification",
                  model="facebook/bart-large-mnli", device=device)

LABELS = {
    "fact-based": "This text cites specific facts, data, or named events.",
    "emotional":  "This text tries to make the reader feel fear or worry.",
    "rebuttal":   "This text directly responds to a claim made by an opponent.",
}

samples = {
    "Haiku, round 3": "My opponent conflates flexibility with safety. Pilot override "
        "capability doesn't prevent accidents, it only permits them. Boeing's 737 MAX "
        "crashes demonstrate this critical difference.",
    "Nova, round 1": "While Airbus's protections are valuable, Boeing's systems also "
        "incorporate robust safeguards and redundancy. Both designs have proven safe "
        "over decades of operation.",
    "Emotional test": "Imagine your own family on board, trusting their lives to a "
        "machine that refuses to listen to the pilot. That should terrify everyone.",
}

for name, text in samples.items():
    result = tagger(text, candidate_labels=list(LABELS.values()),
                    multi_label=True, hypothesis_template="{}")
    short = {v: k for k, v in LABELS.items()}
    print(name)
    for label, score in zip(result["labels"], result["scores"]):
        print(f"   {short[label]:<12} {score:.2f}")
    print()

import time
import torch
from transformers import pipeline

print("MPS available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"

t0 = time.time()
tagger = pipeline("zero-shot-classification",
                  model="facebook/bart-large-mnli", device=device)
print(f"Model ready in {time.time() - t0:.1f}s\n")

LABELS = ["fact-based", "an emotional appeal", "a rebuttal of the opponent"]

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
    t = time.time()
    result = tagger(text, candidate_labels=LABELS, multi_label=True,
                    hypothesis_template="This argument is {}.")
    print(f"{name}  ({time.time() - t:.2f}s)")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"   {label:<28} {score:.2f}")
    print()

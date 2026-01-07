import random

import numpy as np
from scipy.stats import percentileofscore

from granuscore import GranuScore
from nltk.corpus import wordnet as wn

from granuscore.loader import JSONReader


def get_wordnet_noun_lemmas(
    include_multiword: bool = True,
    replace_underscores: bool = True,
) -> list[str]:
    lemmas = set()

    for syn in wn.all_synsets(pos=wn.NOUN):
        for lemma in syn.lemma_names():
            if not include_multiword and "_" in lemma:
                continue
            lemmas.add(lemma)

    out = sorted(lemmas)
    if replace_underscores:
        out = [w.replace("_", " ") for w in out]
    return out

def sample_words_near_percentile(
    texts,
    scores,
    reference_scores,
    target_p: int,
    band: float = 2.5,   # ±2.5 percentile points
    k: int = 10,
    seed: int = 0,
):
    rng = random.Random(seed)

    percentiles = [
        percentileofscore(reference_scores, sc)
        for sc in scores
    ]

    candidates = [
        texts[i]
        for i, p in enumerate(percentiles)
        if abs(p - target_p) <= band
    ]

    if len(candidates) <= k:
        return candidates

    return rng.sample(candidates, k)

noun_texts = get_wordnet_noun_lemmas(include_multiword=True, replace_underscores=True)
scorer = GranuScore()
noun_scores = scorer.predict(noun_texts, encoding_batch_size=128, batch_size=256, show_progress_bar=True, split=False)

np.save("scores-wordnet_nouns.npy", noun_scores)


reference_words = {
    p: sample_words_near_percentile(
        texts=noun_texts,
        scores=noun_scores,
        reference_scores=noun_scores,
        target_p=p,
        band=3,
        k=20,
        seed=42,
    )
    for p in [10, 50, 90]
}

JSONReader().write(f'reference_words_wordnet_nouns.json', reference_words)



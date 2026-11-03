"""Character-preserving windows; no text is silently truncated."""
from __future__ import annotations


def text_windows(tokenizer, text, budget, overlap):
    if budget <= overlap or overlap < 0:
        raise ValueError("Token budget must exceed the non-negative overlap")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= budget:
        yield 0, len(text)
        return
    for index in range(0, len(offsets), budget - overlap):
        stop = min(index + budget, len(offsets))
        begin = 0 if index == 0 else offsets[index][0]
        end = len(text) if stop == len(offsets) else offsets[stop][0]
        yield begin, end
        if stop == len(offsets):
            break

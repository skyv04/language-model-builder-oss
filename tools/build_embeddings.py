"""Regenerate model_builder/ui/data/embeddings.json, the word map in the Embeddings chapter.

Streams GloVe 6B 50d (PDDL 1.0) from the gensim-data mirror and keeps a curated subset. Only the
first part of the 69 MB file is read, because every chosen word is common.

Usage: python tools/build_embeddings.py [glove-wiki-gigaword-50.gz URL]
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"

CATEGORIES = [
    ("animals", "Animals", "dog cat horse cow pig sheep goat chicken duck rabbit mouse lion tiger bear wolf fox deer elephant monkey snake fish bird eagle owl whale dolphin shark frog bee ant puppy kitten horses dogs cats"),
    ("food", "Food & drink", "bread cheese butter milk egg eggs rice pasta soup salad meat beef pork apple banana orange grape lemon cherry strawberry potato tomato carrot onion sugar salt pepper chocolate cake cookie pizza sandwich coffee tea juice wine beer water"),
    ("colors", "Colors", "red blue green yellow black white purple pink brown gray golden silver"),
    ("numbers", "Numbers", "one two three four five six seven eight nine ten twenty hundred thousand million billion first second third"),
    ("people", "People & family", "mother father son daughter brother sister uncle aunt grandmother grandfather wife husband baby child children boy girl man woman men women friend king queen prince princess actor actress mom dad nephew niece"),
    ("jobs", "Jobs", "doctor nurse teacher student lawyer engineer farmer scientist artist writer poet singer pilot soldier police chef judge manager president minister priest programmer"),
    ("body", "Body", "head face eye eyes ear nose mouth hand hands arm leg foot heart brain blood bone skin hair finger teeth"),
    ("places", "Countries & cities", "france paris french germany berlin german italy rome italian spain madrid spanish japan tokyo japanese china beijing chinese russia moscow england london egypt cairo canada ottawa india delhi greece athens brazil mexico kenya nairobi"),
    ("nature", "Nature & weather", "sun moon star stars sky cloud rain snow wind storm river lake sea ocean mountain forest tree trees flower grass rock sand island desert fire ice earth"),
    ("time", "Time & calendar", "monday tuesday wednesday thursday friday saturday sunday january february march april june july august september october november december morning evening night day week month year today tomorrow yesterday summer winter autumn spring"),
    ("travel", "Town & travel", "car bus train plane ship boat bicycle truck road street bridge station airport city town village school church hospital hotel shop market office"),
    ("home", "Home & things", "house home kitchen garden room door window table chair bed cup plate knife spoon book pen paper clock lamp key bag box bottle shirt shoes hat coat dress"),
    ("verbs", "Verbs", "walk walked walking run ran running swim swam swimming eat ate eating go went going see saw seen think thought write wrote written read speak spoke sing sang give gave take took buy bought sell sold make made fly flew"),
    ("adjectives", "Adjectives", "big bigger biggest small smaller smallest good better best bad worse worst happy sad angry afraid hot cold warm fast slow old young new beautiful ugly rich poor strong weak easy hard dark bright quiet loud tall short"),
    ("ideas", "Ideas & feelings", "love hate fear joy anger hope peace war truth freedom justice art science history mathematics language money power idea dream religion"),
    ("hobbies", "Music & sport", "music piano guitar violin drum song dance football soccer tennis golf basketball baseball chess game team player"),
    ("tech", "Technology", "computer software internet phone data code program network screen keyboard robot machine digital online email website"),
    ("function", "Little words", "the a an of and to in on is was he she it they we you i his her their this that with for not but"),
]


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else GLOVE_URL
    target = Path(__file__).resolve().parents[1] / "model_builder" / "ui" / "data" / "embeddings.json"
    wanted: dict[str, int] = {}
    for index, (_, _, words) in enumerate(CATEGORIES):
        for word in words.split():
            if word in wanted:
                raise SystemExit(f"duplicate word {word}")
            wanted[word] = index
    found: dict[str, tuple[int, np.ndarray]] = {}
    request = urllib.request.Request(url, headers={"User-Agent": "lmb-linux-embeddings-builder"})
    with urllib.request.urlopen(request, timeout=60) as response, gzip.GzipFile(fileobj=response) as stream:
        header = stream.readline().decode().split()
        dims = int(header[1])
        for rank, raw in enumerate(stream):
            parts = raw.decode("utf-8", "replace").rstrip("\n").split(" ")
            word = parts[0]
            if word in wanted and word not in found:
                found[word] = (rank, np.asarray(parts[1:], dtype=np.float32))
                if len(found) == len(wanted):
                    break
    missing = [word for word in wanted if word not in found]
    if missing:
        raise SystemExit(f"missing words: {missing}")
    words = list(wanted)
    matrix = np.stack([found[word][1] for word in words]).astype(np.float64)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    centered = unit - unit.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    coords /= np.abs(coords).max(axis=0, keepdims=True)
    data = {
        "source": "GloVe 6B 50d (Wikipedia 2014 + Gigaword 5), Pennington, Socher & Manning, Stanford NLP, 2014",
        "license": "Open Data Commons Public Domain Dedication and License (PDDL) 1.0",
        "url": "https://nlp.stanford.edu/projects/glove/",
        "note": "Curated subset; vectors are unit-normalised and stored as integers (value x 1000). xy = first two principal components of the subset.",
        "dims": dims,
        "scale": 1000,
        "categories": [{"id": cid, "label": label} for cid, label, _ in CATEGORIES],
        "words": [[word, wanted[word], found[word][0], round(float(x), 3), round(float(y), 3)] for word, (x, y) in zip(words, coords)],
        "vectors": [[int(round(value * 1000)) for value in row] for row in unit],
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    print(f"wrote {len(words)} words ({dims} dims) to {target}")


if __name__ == "__main__":
    main()

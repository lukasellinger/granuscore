from granuscore.artifcats import ArtifactSpec

BASE_URL = f"https://github.com/lukasellinger/granuscore/releases/download/v1.0.0"

DEFAULT_FAISS_INDEX = ArtifactSpec(
    name="50k-hit-index.faiss",
    subdir='faiss',
    url=f"{BASE_URL}/50k-hit-index.faiss",
    sha256="e4d37d135fee76d1644a9c4b5311594179a8a53efd50fab324bb98578b8f06df",
)

DEFAULT_ORIGINAL_VECTORS = ArtifactSpec(
    name="50k-hit-original-vectors.npy",
    subdir='faiss',
    url=f"{BASE_URL}/50k-hit-original-vectors.npy",
    sha256="41527089f99dd1bc3eef8d470eb025af2e36e0591cbeae21b66b8f513d5ca6c4",
)

DEFAULT_ANCHORS = ArtifactSpec(
    name="50k-hit-random_anchors_999.npy",
    subdir='faiss',
    url=f"{BASE_URL}/50k-hit-random_anchors_999.npy",
    sha256="92432aabbec27507a71d56595bdd0e8c6ca5aeedafd13ed70af68b01de7e9436",
)

DEFAULT_LGB_MODEL = ArtifactSpec(
    name="50k-hit-random_anchors-999_model.txt",
    subdir='lgb_models',
    url=f"{BASE_URL}/50k-hit-random_anchors-999_model.txt",
    sha256="5aba55b9e13c09d86561b50773d77e5c21c0a7f7c7092473537e28c247d451b9",
)

DEFAULT_NOUN_SCORES = ArtifactSpec(
    name="scores-wordnet_nouns.npy",
    subdir='references',
    url=f"{BASE_URL}/scores-wordnet_nouns.npy",
    sha256="487a2c46afd74af3875058fb2115a5a1dc68d48a7ebde2e7aabeb3b4e2551a53",
)
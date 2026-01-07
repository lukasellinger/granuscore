"""
Build a FAISS index from Wikidata entities with English labels and descriptions.
"""
import re
from typing import Optional, Dict

import faiss
import numpy as np
from datasets import load_dataset, IterableDataset
from hierarchy_transformers import HierarchyTransformer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class WikidataProcessor:
    """Process and extract English entities from Wikidata."""

    ENGLISH_PATTERN = re.compile(r'"en":\{"language":"en","value":"(.*?)"}')

    def __init__(self, include_description: bool = False):
        """Initialize processor with skip counter."""
        self.skipped_count = 0
        self.include_description = include_description

    @staticmethod
    def _fix_unicode(s: str) -> Optional[str]:
        """Fix Unicode encoding issues."""
        if not isinstance(s, str):
            return None
        try:
            return s.encode("utf-8", "replace").decode("utf-8")
        except Exception:
            return None

    @classmethod
    def _safe_unescape(cls, s: str) -> Optional[str]:
        """Safely unescape Unicode escape sequences."""
        if not isinstance(s, str):
            return None
        try:
            return cls._fix_unicode(s.encode("utf-8").decode("unicode_escape"))
        except Exception:
            return cls._fix_unicode(s)

    def clean(self, example: Dict) -> Optional[Dict[str, str]]:
        """
        Extract and clean English label and description from Wikidata example.

        Args:
            example: Dict with 'labels' and 'descriptions' fields

        Returns:
            Dict with 'entity' field if successful, None if skipped
        """
        raw_label = example.get("labels")
        raw_desc = example.get("descriptions")

        # Filter: must both be strings
        if not isinstance(raw_label, str) or not isinstance(raw_desc, str):
            self.skipped_count += 1
            return {'entity': None}

        lbl_match = self.ENGLISH_PATTERN.search(raw_label)
        desc_match = self.ENGLISH_PATTERN.search(raw_desc)

        # No English entry found
        if not lbl_match or not desc_match:
            self.skipped_count += 1
            return {'entity': None}

        # Decode safely
        label = self._safe_unescape(lbl_match.group(1))
        desc = self._safe_unescape(desc_match.group(1))

        # Decoding failed
        if label is None or desc is None:
            self.skipped_count += 1
            return {'entity': None}

        label = self._fix_unicode(label)
        desc = self._fix_unicode(desc)

        # Empty after cleaning
        if not label or not desc:
            self.skipped_count += 1
            return {'entity': None}

        entity = f"{label}"
        if self.include_description:
            entity += f". {desc}"

        return {"entity": entity}


class FAISSIndexBuilder:
    """Build FAISS index from streaming dataset."""

    def __init__(
        self,
        batch_size: int = 256,
        embedding_dim: int = 384,
        model_name: str = 'Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun',
        use_hierarchy_transformer: bool = True,
    ):
        """
        Initialize index builder.

        Args:
            model_name: Name of sentence transformer model
            batch_size: Batch size for encoding
            embedding_dim: Dimension of embeddings (must match model output)
        """
        if use_hierarchy_transformer:
            self.model = HierarchyTransformer.from_pretrained(model_name)
        else:
            self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.include_descriptions = include_descriptions
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.original_vectors = None

    def build_from_dataset(
        self,
        dataset: IterableDataset,
        output_path: str = "full_index.faiss",
    ):
        """
        Build FAISS index from streaming dataset.

        Args:
            dataset: Iterable dataset with batched entities
            output_path: Path to save the index

        Returns:
            Built FAISS index
        """
        print(f"Building FAISS index with batch size {self.batch_size}...")

        all_original = []
        for batch in tqdm(dataset, desc="Indexing batches"):
            # 1. get UNnormalized embeddings
            embeddings = self.model.encode(
                batch["entity"],
                normalize_embeddings=False
            ).astype(np.float32)

            # store originals
            all_original.append(embeddings)

            # 2. normalize a copy for FAISS
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            normalized = embeddings / norms

            # 3. add to FAISS
            self.index.add(normalized)

        self.original_vectors = np.vstack(all_original)

        for batch in tqdm(dataset, desc="Indexing batches"):
            embeddings = self.model.encode(batch["entity"], normalize_embeddings=True)
            self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors")
        print(f"Saving index to {output_path}...")
        faiss.write_index(self.index, output_path)

        return self.index, self.original_vectors


def main(output_path, model, use_hierarchy_transformer, include_description, size=None, random=False):
    """Main execution function."""
    # Configuration
    BATCH_SIZE = 256
    DATASET_NAME = "philippesaade/wikidata"
    COLUMNS_TO_REMOVE = ["labels", "descriptions", "aliases", "sitelinks", "claims"]

    # Load streaming dataset
    print(f"Loading dataset: {DATASET_NAME}")
    streaming_ds = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    # Process and clean entities
    print("Cleaning and processing entities...")
    processor = WikidataProcessor(include_description=include_description)
    streaming_ds = streaming_ds.map(
        processor.clean,
        remove_columns=COLUMNS_TO_REMOVE
    )

    # Filter out None values (skipped entries)
    streaming_ds = streaming_ds.filter(lambda x: x["entity"] is not None)

    if random:
        streaming_ds.shuffle(
            buffer_size=10_000,
            seed=42
        )

    if size:
        batched_dataset = streaming_ds.take(size).batch(batch_size=BATCH_SIZE)
    else:
        batched_dataset = streaming_ds.batch(batch_size=BATCH_SIZE)

    # Build FAISS index
    builder = FAISSIndexBuilder(batch_size=BATCH_SIZE, model_name=model, use_hierarchy_transformer=use_hierarchy_transformer)
    index, original_vectors = builder.build_from_dataset(batched_dataset, output_path)

    np.save("50k-hit-original-vectors.npy", original_vectors.astype(np.float32))

    assert original_vectors.shape[0] == index.ntotal

    print(f"\nSkipped {processor.skipped_count} entries during cleaning")
    print("Done!")
    return index


if __name__ == "__main__":
    include_descriptions = False
    random = True
    output_path = "50k-hit-index.faiss"
    size = 50000

    model = 'Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun'
    use_hierarchy_transformer = True
    #model = 'sentence-transformers/all-MiniLM-L6-v2'
    #use_hierarchy_transformer = False
    main(output_path, model, use_hierarchy_transformer, include_descriptions, size, random)

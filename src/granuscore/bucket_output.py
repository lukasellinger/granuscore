import numpy as np
from scipy.stats import percentileofscore


class PercentileOutput:
    def __init__(self, path):
        self.reference_scores = np.load(path)
        self.percentiles: dict[float, float] | None = None

    def _compute_percentiles(self):
        self.percentiles = {
            p: float(np.percentile(self.reference_scores, p))
            for p in [33.33, 66.66]
        }

    def score_to_percentile(self, scores: float | list[float] | np.ndarray):
        scores = np.asarray(scores)
        return percentileofscore(self.reference_scores, scores)

    def percentile_to_score(self, percentiles: float | list[float] | np.ndarray):
        is_scalar = np.isscalar(percentiles)
        arr = np.asarray([percentiles] if is_scalar else percentiles, dtype=float)

        scores = np.percentile(self.reference_scores, arr)

        if is_scalar:
            return float(scores[0])
        return scores

    def bucket(self, scores: float | list[float] | np.ndarray) -> str:
        """
        Assign a granularity bucket based on score-space percentiles.

        Args:
            scores:
                Granuscore percentile value.
        Returns:
            One of: "low", "medium", "high"
        """
        if self.percentiles is None:
            self._compute_percentiles()

        q1 = 33.33
        q2 = 66.66

        is_scalar = np.isscalar(scores)
        arr = np.asarray([scores] if is_scalar else scores, dtype=float)

        labels = np.where(arr < q1, "low", np.where(arr < q2, "medium", "high"))

        if is_scalar:
            return str(labels[0])
        return labels.tolist()

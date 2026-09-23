"""Train-only GT labeling and strict RF bundle compatibility."""
from pathlib import Path
import hashlib
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .features import FEATURE_NAMES

FORMAT = 'radarocc-autoware-gm2019-occupancy-first-rf-v1'


def annotation_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_target(candidate, gt, positive_fraction=.2, negative_fraction=.05):
    labels = gt[tuple(candidate['voxels'].T)]
    labels = labels[labels != 255]
    if not len(labels):
        return None, 'ignored'
    foreground = np.mean(labels == 2)
    background = np.mean(labels == 1)
    if foreground >= positive_fraction:
        return 1, 'foreground'
    if foreground <= negative_fraction and background >= positive_fraction:
        return 0, 'background'
    # Do not train empty space as occupied Background.
    return None, 'free_or_ambiguous'


class RandomForest:
    def __init__(self, estimator, metadata):
        self.estimator, self.metadata = estimator, metadata

    def foreground_probabilities(self, features):
        return self.estimator.predict_proba(features)[:, list(self.estimator.classes_).index(1)]

    @classmethod
    def load(cls, path, config):
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get('format') != FORMAT:
            raise ValueError('Requires a tradition_real RF model; old tradition checkpoints are incompatible')
        if tuple(bundle.get('feature_names', ())) != FEATURE_NAMES:
            raise ValueError('RF feature names/order mismatch')
        if bundle.get('config') != config.signature():
            raise ValueError('RF pipeline configuration differs from this run; use its saved configuration')
        estimator = bundle['estimator']
        if estimator.n_features_in_ != 42 or set(estimator.classes_) != {0, 1}:
            raise ValueError('RF must accept 42 features and classes {0,1}')
        return cls(estimator, bundle['metadata'])


def train(features, targets, path, config, metadata, n_estimators=200, seed=13,
          max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1):
    x, y = np.asarray(features, np.float64), np.asarray(targets, np.int8)
    if x.ndim != 2 or x.shape[1] != 42 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,42] training features and N targets')
    if set(y) != {0, 1}:
        raise ValueError('Training proposals must include both occupied Background and Foreground')
    estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight=class_weight, n_jobs=n_jobs,
        random_state=seed, oob_score=True)
    estimator.fit(x, y)
    metadata = {**metadata, 'sample_count': len(y), 'background_samples': int((y == 0).sum()),
                'foreground_samples': int((y == 1).sum()), 'oob_score': float(estimator.oob_score_),
                'n_estimators': n_estimators, 'seed': seed, 'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf, 'class_weight': class_weight, 'n_jobs': n_jobs}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'format': FORMAT, 'feature_names': FEATURE_NAMES, 'config': config.signature(),
                 'estimator': estimator, 'metadata': metadata}, path)
    return metadata

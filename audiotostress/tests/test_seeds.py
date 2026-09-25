import random

import numpy as np
import pytest

from ukstress.provenance.seeds import derive_seed, seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences() -> None:
    seed_everything(123)
    first = (random.random(), np.random.random())
    seed_everything(123)
    second = (random.random(), np.random.random())

    assert first == second


def test_derived_seeds_are_stable_and_namespaced() -> None:
    assert derive_seed(17, namespace="loader", index=2) == derive_seed(
        17, namespace="loader", index=2
    )
    assert derive_seed(17, namespace="loader", index=2) != derive_seed(
        17, namespace="split", index=2
    )


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_invalid_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must"):
        seed_everything(seed)

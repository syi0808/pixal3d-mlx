import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest
from safetensors.numpy import save_file

from pixalmlx.weight_loader import load_weights


@pytest.mark.parametrize("dtype", [None, mx.float32])
def test_streaming_weights_preserve_legacy_rounding(tmp_path, dtype):
    path = tmp_path / "tiny.safetensors"
    rng = np.random.default_rng(2)
    weights = {"weight": rng.standard_normal((4, 8)).astype(np.float32),
               "bias": rng.standard_normal((4,)).astype(np.float32)}
    save_file(weights, path)
    expected = nn.Linear(8, 4)
    load_weights(expected, str(path), verbose=False)
    if dtype is not None:
        expected.apply(lambda value: value.astype(dtype))
    actual = nn.Linear(8, 4)
    assert load_weights(actual, str(path), verbose=False, streaming=True, target_dtype=dtype) == []
    for (name, reference), (actual_name, result) in zip(mlx.utils.tree_flatten(expected.parameters()),
                                                      mlx.utils.tree_flatten(actual.parameters())):
        assert name == actual_name
        np.testing.assert_array_equal(np.array(result), np.array(reference))
        assert result.dtype == reference.dtype


def test_streaming_weights_reject_missing_and_wrong_shapes(tmp_path):
    path = tmp_path / "tiny.safetensors"
    save_file({"weight": np.ones((4, 8), np.float32)}, path)
    with pytest.raises(ValueError, match="Missing"):
        load_weights(nn.Linear(8, 4), str(path), streaming=True)
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_weights(nn.Linear(8, 5), str(path), streaming=True)

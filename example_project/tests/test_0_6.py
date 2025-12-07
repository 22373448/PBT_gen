import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(value=st.text())
def test_encode_is_pure_and_side_effect_free(value):
    encoder = Encoder()
    result1 = encoder.encode(value)
    result2 = encoder.encode(value)
    assert result1 == result2
    assert result1 is not value
    assert result2 is not value
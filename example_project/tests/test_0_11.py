import hypothesis.strategies as st
from hypothesis import given, assume
from pkg.module_a import Encoder

@given(st.text())
def test_encode_input_non_null(value):
    encoder = Encoder()
    assume(value is not None)
    result = encoder.encode(value)
    assert isinstance(result, str)
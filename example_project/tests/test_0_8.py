import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(value=st.text())
def test_encode_returns_string(value):
    encoder = Encoder()
    result = encoder.encode(value)
    assert isinstance(result, str)
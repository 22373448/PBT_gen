import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(value=st.text())
def test_encode_returns_uppercase_version_of_input(value):
    encoder = Encoder()
    result = encoder.encode(value)
    assert result == value.upper()
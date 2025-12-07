import pytest
from hypothesis import given, strategies as st
from pkg.module_a import Encoder

@given(st.text())
def test_encode_input_must_be_string(value):
    encoder = Encoder()
    result = encoder.encode(value)
    assert isinstance(result, str)
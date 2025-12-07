import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(st.text())
def test_encode_is_idempotent(value):
    encoder = Encoder()
    first_encode = encoder.encode(value)
    second_encode = encoder.encode(first_encode)
    assert first_encode == second_encode
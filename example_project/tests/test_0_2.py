from hypothesis import given, strategies as st
from pkg.module_a import Encoder

@given(st.text())
def test_encode_output_length_equals_input_length(value: str) -> None:
    encoder = Encoder()
    result = encoder.encode(value)
    assert len(result) == len(value)
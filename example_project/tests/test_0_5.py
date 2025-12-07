import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(st.text())
def test_encode_output_is_entirely_uppercase(value: str) -> None:
    encoder = Encoder()
    result = encoder.encode(value)
    assert result.isupper() or not any(c.isalpha() for c in result)
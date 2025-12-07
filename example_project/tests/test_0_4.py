import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(value=st.text())
def test_encode_output_contains_only_uppercase_or_original(value):
    encoder = Encoder()
    result = encoder.encode(value)
    for original_char, result_char in zip(value, result):
        if original_char.isalpha():
            assert result_char.isupper()
        else:
            assert result_char == original_char
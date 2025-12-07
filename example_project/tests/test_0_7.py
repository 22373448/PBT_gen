import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(st.text())
def test_encode_output_characters_uppercase_or_non_alpha(value):
    encoder = Encoder()
    result = encoder.encode(value)
    for c in result:
        assert c.isupper() or not c.isalpha()
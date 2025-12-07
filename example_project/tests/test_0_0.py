import hypothesis.strategies as st
from hypothesis import given
from pkg.module_a import Encoder

@given(s1=st.text(), s2=st.text())
def test_encode_is_deterministic_for_equal_strings(s1, s2):
    encoder = Encoder()
    if s1 == s2:
        assert encoder.encode(s1) == encoder.encode(s2)
import pytest
from hypothesis import given, strategies as st
from pkg.module_a import Encoder

class TestEncoder:
    def setup_method(self):
        self.encoder = Encoder()

    @given(st.text())
    def test_output_length_equals_input_length(self, value):
        result = self.encoder.encode(value)
        assert len(result) == len(value)

    @given(st.text())
    def test_output_contains_only_uppercase(self, value):
        result = self.encoder.encode(value)
        assert result == result.upper()

    @given(st.text())
    def test_preserves_non_alphabetic_characters(self, value):
        result = self.encoder.encode(value)
        for i, char in enumerate(value):
            if not char.isalpha():
                assert result[i] == char

    @given(st.text())
    def test_output_is_uppercase_version(self, value):
        result = self.encoder.encode(value)
        assert result == value.upper()

    @given(st.text())
    def test_output_is_always_string(self, value):
        result = self.encoder.encode(value)
        assert isinstance(result, str)

    def test_empty_string(self):
        result = self.encoder.encode('')
        assert result == ''

    @given(st.text(min_size=1))
    def test_pure_function(self, value):
        encoder = Encoder()
        result1 = encoder.encode(value)
        result2 = encoder.encode(value)
        assert result1 == result2

    @given(st.none())
    def test_none_input_raises_error(self, value):
        with pytest.raises(Exception):
            self.encoder.encode(value)
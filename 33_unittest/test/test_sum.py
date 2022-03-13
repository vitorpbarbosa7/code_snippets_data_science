from src.sums import new_sum
import pytest

def test_new_sum():
    array_test = [1,2,3]
    assert new_sum(array_test) == 6    
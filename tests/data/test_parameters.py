from   project.data.parameters import is_biolevel

def test_is_biological_level():
    assert is_biolevel("alt")
    assert is_biolevel("amylase_")
    assert is_biolevel("ast_32")
    assert is_biolevel("creatinine_max")
    assert not is_biolevel("altitude")

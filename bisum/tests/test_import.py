# testing importing

import types

import bisum

def test_top_level_import_bisum():
    try:
        from bisum import bisum
    except ImportError:                                            # pragma: nocover
        raise AssertionError("Failed to import bisum from bisum")  # pragma: nocover

def test_top_level_bisum_function():
    # check that this is really the function
    if not isinstance(bisum.bisum, types.FunctionType):            # pragma: nocover
        raise AssertionError("bisum.bisum() is not a function")    # pragma: nocover


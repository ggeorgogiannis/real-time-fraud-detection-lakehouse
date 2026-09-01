from importlib import import_module


def test_package_is_importable() -> None:
    package = import_module("fraud_lakehouse")

    assert package.__name__ == "fraud_lakehouse"

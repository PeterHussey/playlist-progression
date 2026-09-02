import pytest


def pytest_configure(config):
    """Register network marker and skip network tests by default."""
    config.addinivalue_line("markers", "network: mark test as requiring network")
    # By default, skip tests marked with 'network'
    config.option.markexpr = "not network"
def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")
    parser.addoption("--slim", action="store_true",
                     help="run base + current tests")


_slim = False
_slim_reason = ""


def slim():
    return _slim


# def pytest_configure(config):
#     global _slim
#     global _slim_reason
#     _slim = config.getoption("--slim")

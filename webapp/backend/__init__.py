"""Searchlight backend package."""

__all__ = ["app", "create_app"]


def create_app(*args, **kwargs):
    from .main import create_app as _create_app

    return _create_app(*args, **kwargs)


def __getattr__(name: str):
    if name == "app":
        from .main import app

        return app
    raise AttributeError(name)

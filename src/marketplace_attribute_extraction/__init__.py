"""Marketplace attribute extraction package."""

from __future__ import annotations

__all__ = ["ArtifactPaths", "ProductAttributeExtractor"]


def __getattr__(name: str):
    if name in __all__:
        from .inference import ArtifactPaths, ProductAttributeExtractor
        return {
            "ArtifactPaths": ArtifactPaths,
            "ProductAttributeExtractor": ProductAttributeExtractor,
        }[name]
    raise AttributeError(name)

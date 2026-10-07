from .temporal_plugin import TemporalPlugin


def build_temporal_plugin(cfg):
    """Build the DG-STF plugin without introducing an external registry."""
    if cfg is None:
        return None

    cfg = dict(cfg)
    plugin_type = str(
        cfg.pop("type", "TemporalPlugin")
    ).lower()

    if plugin_type not in (
        "temporalplugin",
        "temporal_plugin",
        "identity",
    ):
        raise KeyError(
            f"Unsupported temporal plugin type: {plugin_type}"
        )

    if plugin_type == "identity":
        cfg.setdefault("mode", "identity")

    return TemporalPlugin(**cfg)

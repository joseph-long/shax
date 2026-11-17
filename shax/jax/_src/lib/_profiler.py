"""Stand-in for jaxlib._profiler: shax has no profiler."""
from shax.jax._src.lib import _unsupported as _u


def register_plugin_profiler(*args, **kwargs) -> None:
  pass


class TraceMe:
  """No-op trace annotation (jaxlib's TraceMe is a C++ context manager)."""

  def __init__(self, name, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def set_metadata(self, **kwargs):
    pass

  @staticmethod
  def is_enabled() -> bool:
    return False


ProfileOptions = _u.cls("_profiler.ProfileOptions")
ProfilerServer = _u.cls("_profiler.ProfilerServer")
ProfilerSession = _u.cls("_profiler.ProfilerSession")
for _name in ("aggregate_profiled_instructions", "export_to_xprof",
              "get_fdo_profile", "register_subprocess", "start_server"):
  globals()[_name] = _u.function(f"_profiler.{_name}")


def set_metadata(*args, **kwargs) -> None:
  pass


def clear_metadata(*args, **kwargs) -> None:
  pass

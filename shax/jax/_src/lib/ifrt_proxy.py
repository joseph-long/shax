"""Stand-in for jaxlib._ifrt_proxy: shax has no remote-runtime support."""


class ClientConnectionOptions:
  pass


def get_client(target, options):
  raise RuntimeError("shax does not support IFRT proxy backends")

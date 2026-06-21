from __future__ import annotations

from enum import IntEnum
from typing import Any

import cloudpickle
import msgspec
from msgspec import msgpack

CUSTOM_TYPE_CLOUDPICKLE = -2


class MsgpackEncoder:
    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: Any) -> list[bytes]:
        """Encode object to msgpack bytes list (ZMQ multipart frames)."""
        return [self.encoder.encode(obj)]

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, IntEnum):
            return obj.value
        return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))


class MsgpackDecoder:
    def __init__(self, t: Any = None):
        args = () if t is None else (t,)
        self._default_type = t
        self.decoder = msgpack.Decoder(*args, ext_hook=self.ext_hook, dec_hook=self.dec_hook)

    def decode(self, bufs: list[bytes], t: Any = None) -> Any:
        """Decode from ZMQ multipart frames (bufs[0] = msgpack payload).

        If ``t`` is provided, decode as that type. Otherwise uses the type
        given to the constructor, or falls back to untyped decoding.
        """
        buf = bufs[0]
        if t is not None:
            return msgpack.decode(buf, type=t, ext_hook=self.ext_hook, dec_hook=self.dec_hook)
        return self.decoder.decode(buf)

    def dec_hook(self, t: type, obj: Any) -> Any:
        if t is Any or t is object:
            return obj
        if isinstance(t, type) and issubclass(t, msgspec.Struct):
            return self._rebuild_struct(t, obj)
        raise NotImplementedError(t)

    def _rebuild_struct(self, t: type, obj: Any) -> Any:
        fields = msgspec.structs.fields(t)
        is_array_like = msgspec.structs.get_def(t).array_like
        if is_array_like:
            if not isinstance(obj, (list, tuple)):
                obj = list(obj)
            defaults = [
                f.default if f.default is not msgspec.UNSET else (f.default_factory() if f.default_factory is not msgspec.UNSET else None) for f in fields
            ]
            filled = list(obj)
            while len(filled) < len(defaults):
                filled.append(defaults[len(filled)])
            return t(*filled)
        else:
            return t(**obj)

    def ext_hook(self, code: int, data: bytes) -> Any:
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)
        raise ValueError(f"Unknown ext type: {code}")

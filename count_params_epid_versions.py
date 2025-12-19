import importlib
import inspect

import torch


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_trainable(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def instantiate(cls, **preferred_kwargs):
    sig = inspect.signature(cls)
    kwargs = {}
    for name in sig.parameters:
        if name in preferred_kwargs:
            kwargs[name] = preferred_kwargs[name]
    return cls(**kwargs)


def report(version: str, modname: str):
    mod = importlib.import_module(modname)
    enc_cls = getattr(mod, "EncoderUNet")
    dec_cls = getattr(mod, "UNetDecoder")

    preferred = dict(vector_dim=104, scalar_count=5)

    enc = instantiate(enc_cls, **preferred).cpu()
    dec = instantiate(dec_cls, **preferred).cpu()

    print(version)
    print(f"  EncoderUNet: total={count_params(enc):,} trainable={count_trainable(enc):,}")
    print(f"  UNetDecoder: total={count_params(dec):,} trainable={count_trainable(dec):,}")
    print(
        f"  Combined:    total={count_params(enc) + count_params(dec):,} "
        f"trainable={count_trainable(enc) + count_trainable(dec):,}"
    )
    print()


def main() -> None:
    for version, modname in [
        ("v4", "generate_and_train_EPID_v4"),
        ("v5", "generate_and_train_EPID_v5"),
        ("v6", "generate_and_train_EPID_v6"),
    ]:
        report(version, modname)


if __name__ == "__main__":
    main()

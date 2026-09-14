import os

import torch

CKPT_NAMES = ("last.pt", "best.pt")
STATE_KEYS = ("model_state_dict", "model")


def _weight_path(path):
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        for name in CKPT_NAMES:
            candidate = os.path.join(path, name)

            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(f"Weights not found: {path}")


def _load_state(path):
    state = torch.load(_weight_path(path), map_location="cpu", weights_only=False)

    for key in STATE_KEYS:
        if key in state:
            state = state[key]
            break

    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state.items()
    }


def load_weights(model, path):
    try:
        model.load_state_dict(_load_state(path))
    except RuntimeError as error:
        raise RuntimeError(
            f"{path} does not match {type(model).__name__}. The checkpoint was "
            "most likely trained with a different encoder shape (num_layers or "
            "tcn_channels) or by an older version of the code."
            f"\n\n{error}"
        ) from None


VISUAL_PREFIX_MAP = (
    ("tcn.tcn_trunk.network.", "tcn.network."),
    ("frontend3D.", "frontend.frontend3D."),
    ("trunk.0.", "frontend.trunk.features."),
    ("trunk.1.", "frontend.trunk.conv_last."),
)

DISCARDED_PREFIXES = ("tcn.tcn_output.",)


def visual_state(path):
    state = {}

    for key, value in _load_state(path).items():
        if key.startswith(DISCARDED_PREFIXES) or "net" in key.split(".")[:-1]:
            continue

        for old, new in VISUAL_PREFIX_MAP:
            if key.startswith(old):
                state[new + key.removeprefix(old)] = value
                break
        else:
            raise ValueError(f"Unrecognised key in visual checkpoint: {key}")

    if not state:
        raise ValueError(f"No frontend3D./trunk./tcn. weights found in {path}")

    return state


def load_visual_pretrained(encoder, path):
    state = visual_state(path)
    missing, unexpected = encoder.load_state_dict(state, strict=False)

    pretrained_blocks = {
        key.split(".")[2] for key in state if key.startswith("tcn.network.")
    }

    def is_fresh(key):
        parts = key.split(".")

        if parts[0] == "projection":
            return True

        return parts[:2] == ["tcn", "network"] and parts[2] not in pretrained_blocks

    missing = [key for key in missing if not is_fresh(key)]

    if missing or unexpected:
        raise ValueError(
            f"{path} does not match the visual encoder. Check width_mult and "
            "relu_type: the released checkpoints use PReLU, and asking for "
            "relu_type='relu' drops every activation weight."
            f"\n  missing={missing[:5]}\n  unexpected={unexpected[:5]}"
        )

    print(
        f"loaded {len(pretrained_blocks)}/{len(encoder.tcn.network)} temporal "
        f"blocks and the visual frontend from {path}"
    )

    return encoder

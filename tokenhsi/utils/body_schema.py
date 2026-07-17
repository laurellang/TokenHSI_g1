import torch


def _as_float3(value):
    if value is None:
        return torch.zeros(3, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def build_asset_to_motion_body_mapping(
    asset_body_names,
    asset_body_parent_names,
    motion_body_names,
    asset_body_rest_pos=None,
):
    motion_body_index = {name: idx for idx, name in enumerate(motion_body_names)}
    asset_body_rest_pos = asset_body_rest_pos or {}
    body_ids = []
    pos_offsets = []

    for body_name in asset_body_names:
        curr_name = body_name
        visited = set()
        while curr_name not in motion_body_index:
            visited.add(curr_name)
            parent_name = asset_body_parent_names.get(curr_name)
            if parent_name is None or parent_name in visited:
                curr_name = motion_body_names[0]
                break
            curr_name = parent_name

        body_ids.append(motion_body_index[curr_name])
        body_rest = _as_float3(asset_body_rest_pos.get(body_name))
        mapped_rest = _as_float3(asset_body_rest_pos.get(curr_name))
        pos_offsets.append(body_rest - mapped_rest)

    return torch.tensor(body_ids, dtype=torch.long), torch.stack(pos_offsets, dim=0)


def map_motion_body_state_to_asset(body_state, body_ids, pos_offsets=None):
    asset_state = body_state.index_select(1, body_ids)
    if pos_offsets is not None:
        asset_state = asset_state.clone()
        asset_state[..., 0:3] += pos_offsets.to(device=asset_state.device, dtype=asset_state.dtype).unsqueeze(0)
    return asset_state

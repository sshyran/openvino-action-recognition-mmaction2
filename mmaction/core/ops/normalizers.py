

def balance_losses(new_values, meta, gamma=0.9, eps=1e-6):
    if len(new_values) == 0:
        return []

    sum_smoothed_values = 0.0
    for key, new_value in new_values:
        scalar_new_value = float(new_value) if isinstance(new_value, (int, float)) else float(new_value.item())

        if key in meta:
            smoothed_value = gamma * meta[key] + (1.0 - gamma) * scalar_new_value
        else:
            smoothed_value = scalar_new_value

        meta[key] = smoothed_value
        sum_smoothed_values += smoothed_value

    trg_value = sum_smoothed_values / float(len(new_values))
    weights = {key: trg_value / meta[key] if meta[key] > eps else 1.0 for key, _ in new_values}
    weighted_values = [weights[key] * value for key, value in new_values]

    return weighted_values

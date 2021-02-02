

def balance_losses(new_values, meta, gamma=0.9):
    if len(new_values) == 0:
        return []

    sum_smoothed_values = 0.0
    for key, new_value in new_values:
        if key in meta:
            old_value = meta[key]
            smoothed_value = gamma * old_value + (1.0 - gamma) * new_value.item()
        else:
            smoothed_value = new_value.item()

        meta[key] = smoothed_value
        sum_smoothed_values += smoothed_value

    trg_value = sum_smoothed_values / float(len(new_values))
    weighted_values = [float(trg_value / meta[key]) * value for key, value in new_values]

    return weighted_values

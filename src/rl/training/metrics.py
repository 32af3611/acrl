def r2(y_pred, y_true):
    ss_res = (y_true - y_pred) ** 2
    ss_tot = (y_true - y_true.mean()) ** 2
    coeff = 1 - (ss_res.sum() / ss_tot.sum())
    return coeff

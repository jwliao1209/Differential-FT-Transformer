from sklearn.metrics import accuracy_score, r2_score


cls_eval_funs = {
    'accuracy': accuracy_score,
}
reg_eval_funs = {
    'r2': r2_score,
}

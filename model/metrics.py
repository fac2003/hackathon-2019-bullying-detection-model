from fastai.imports import torch, Tensor
from fastai.torch_core import Rank0Tensor


def F1(y_pred:Tensor, y_true:Tensor, eps:float=1e-9)->Rank0Tensor:
    "Computes the f_beta between `preds` and `targets`"
    beta2=1
    y_pred = y_pred.argmax(dim=1)
    y_true = y_true.float()

    TP = ((y_pred > 0.5) * (y_true == 1)).float().sum()
    total_y_pred = (y_pred == 1).float().sum()
    total_y_true = (y_true == 1).float().sum()

    prec = TP/(total_y_pred+eps)
    rec = TP/(total_y_true+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()

if __name__ == "__main__":
    print(F1(torch.tensor([[0,1],[1,0]]), y_true=torch.tensor([[1],[0]])))
    print(F1(torch.tensor([[0,0.55],[0.6,0]]), y_true=torch.tensor([[1],[0]])))
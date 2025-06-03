import numpy as np
import torch as t


  
  

def ER(out, score):
    if t.norm(score) == 0:
        return t.norm(out-score)
    else:
        return t.norm(out-score)/t.norm(score)

  
  
    
def RSE(out, score):
    if t.sum((score-score.mean()).mul(score-score.mean())) == 0:
        return t.sqrt(t.sum((out-score).mul(out-score)))
    else:
        return t.sqrt(t.sum((out-score).mul(out-score))/t.sum((score-score.mean()).mul(score-score.mean())))



def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


  
  

def MAE(out, score):
    count = out.numel()
    return t.sum(t.abs(out-score))/count


  
  

def MSE(out, score):
    count = out.numel()
    return t.sum((out - score).pow(2)) / count


  
  
    
def RMSE(out, score):
    count = out.numel()
    return t.sqrt(t.sum((out-score).mul(out-score))/count)


  
  
    
def MAPE(out, score):
    count = out.numel()
    return (t.sum(t.abs((out-score)/(score+0.000001)))/count)


  
  

def MSPE(out, score):
    count = out.numel()
    return (t.sum(((out - score) / (score+ 0.000001)).pow(2)) / count) 


def metric(pred, true):
    er = ER(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rse = RSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return er, mae, mse,rse, rmse, mape, mspe

import numpy as np
import torch
  
import datetime
import pickle
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot as plt


def train(args,test_loader,model, config, train_loader, valid_loader=None, nsample=100, scaler=1, valid_epoch_interval=20, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)    
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])    
    p2 = int(0.9 * config["epochs"])    
    missing_type = config["missing_type"]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(    
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10    
      
    loss_s = []    
    hproportion_s = []    

    hist_proportion = config["hist_proportion"]
    print("\n train is started, and hist_proportion = ", hist_proportion, ",the config.hist_proportion = ", config["hist_proportion"])
    
    for epoch_no in range(config["epochs"]):    
        loss_sum = 0
          
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:    
            for batch_no, train_batch in enumerate(it, start=1):    
                optimizer.zero_grad()    

                loss, index = model(train_batch, missing_type, is_train=1)    
                loss.backward()    
                  
                loss_sum += loss.item()    
                optimizer.step()    
                if loss_sum != loss_sum:    
                    print("avg_loss is NAN, index = ", index)
                    lss+=1    
                it.set_postfix(
                    ordered_dict={
                        "avg_loss": loss_sum / batch_no,
                        "loss_sum": loss_sum,
                        "batch_no": batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()    

        with open(
                foldername + "/result_nsample" + str(nsample) + ".txt", "a"
            ) as f:
                print("epoch:", epoch_no, file=f)
                print("loss:", loss, file=f)

        evaluate(args, model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      

      
      
      
      
      
      

def evaluate(args, model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        hist_proportion = args.histproportion
        print("\n evaluate is started, and hist_proportion = ", hist_proportion, ",the args.histproportion = ", args.histproportion)
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, hist_proportion, nsample)

                samples, samples_final, c_target, eval_points, observed_points, observed_time = output    
                  
                samples = samples.permute(0, 1, 3, 2)    
                samples_final = samples_final.permute(0, 2, 1)
                c_target = c_target.permute(0, 2, 1)    
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                  
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_final - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_final - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()
                print(f'evalpoints_total: {evalpoints_total}')
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)    
                all_evalpoint = torch.cat(all_evalpoint, dim=0)    
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)    

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,    
                        mean_scaler,    
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

              
              
              
              
              
              
              
              
              
              
              
              
              
              
              

            RMSE = np.sqrt(mse_total / evalpoints_total)
            MAE = mae_total / evalpoints_total
            with open(
                foldername + "/result_nsample" + str(nsample) + ".txt", "a"
            ) as f:
                print(args)
                print(args, file=f)
                print("testmissingratio:", args.testmissingratio, file=f)
                print("RMSE:", RMSE, file=f)
                print("MAE:", MAE, file=f)
                print("CRPS:", CRPS, file=f)
                print("CRPS_sum:", CRPS_sum, file=f)
                print("histproportion:", args.histproportion, file=f)
                print('\n',file=f)


            with open(
                    "./save/" + args.dataset + "_" + str(args.testmissingratio) + "_" + datetime.datetime.now().strftime("%Y%m%d") + ".txt", "a"
            ) as f:
                print(args)
                print("{}\t{}\t{}\t{}\t{}\t{}".format(args.epochs, args.matchingtimes, args.histproportion, "%.4f" % RMSE, "%.4f" % MAE, "%.4f" % CRPS), file=f)

def train_add_history(args,test_loader,model, config, train_loader, valid_loader=None, nsample=100, scaler=1, valid_epoch_interval=20, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)    
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])    
    p2 = int(0.9 * config["epochs"])    
    missing_type = config["missing_type"]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(    
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10    
      
    loss_s = []    
    hproportion_s = []    

    hist_proportion = config["hist_proportion"]
    print("\n train is started, and hist_proportion = ", hist_proportion, ",the config.hist_proportion = ", config["hist_proportion"])
    
    for epoch_no in range(config["epochs"]):    
        loss_sum = 0
          
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:    
            for batch_no, train_batch in enumerate(it, start=1):    
                optimizer.zero_grad()    

                loss, index = model(train_batch, missing_type, is_train=1)    
                loss.backward()    
                  
                loss_sum += loss.item()    
                optimizer.step()    
                if loss_sum != loss_sum:    
                    print("avg_loss is NAN, index = ", index)
                    lss+=1    
                it.set_postfix(
                    ordered_dict={
                        "avg_loss": loss_sum / batch_no,
                        "loss_sum": loss_sum,
                        "batch_no": batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()    

        with open(
                foldername + "/result_nsample" + str(nsample) + ".txt", "a"
            ) as f:
                print("epoch:", epoch_no, file=f)
                print("loss:", loss, file=f)

        evaluate(args, model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

    


def painting_loss(epochs, valid_epoch_interval, loss_s):    
      
      
      
      
      

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs, valid_epoch_interval), loss_s)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()    
    print("Done!")    

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)    
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)    
    target = target * scaler + mean_scaler
    target = target.sum(-1)    
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


  
def calculate_er(pred, true):
    return torch.sum(pred - true)

def calculate_mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def calculate_mse(pred, true):
    return torch.mean((pred - true) ** 2)

def calculate_rse(pred, true):
    return torch.sum((pred - true) ** 2) / torch.sum((true - torch.mean(true)) ** 2)

def calculate_rmse(pred, true):
    return torch.sqrt(calculate_mse(pred, true))

def calculate_mape(pred, true):
    return torch.mean(torch.abs((pred - true) / true)) * 100

def calculate_mspe(pred, true):
    return torch.mean(((pred - true) / true) ** 2) * 100
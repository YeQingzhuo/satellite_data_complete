 

 
 

import pickle
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset



def get_similarity_score(observed, history, strategy):   
    x = observed.copy()
    y = history.copy()
    value_num = 0
    similarity = 1.0e+8
    for i in range(len(x)):
        if x[i] == 0 or y[i] == 0:
            x[i] = y[i] = 0
        else:
            value_num += 1
    if value_num > 0:
        if strategy == "L2":
            similarity = np.sqrt(np.sum((x - y) ** 2)) / value_num   
        elif strategy == "mht":
            similarity = np.sum(np.abs(x - y)) / value_num   
        elif strategy == "dtw":   
            m, n = len(x), len(y)
            dp = np.zeros((m + 1, n + 1))
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = abs(x[i - 1] - y[j - 1])
                    dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            similarity = 2 ** (1 / (dp[m][n] + 1)) - 1
    if similarity == 0:   
        similarity = 1.0e-2

    return similarity


def matching_hist_segment(observed_data, history_data, dist_strategy, section_num):   
    t1 = 0   
    dist = np.zeros(len(observed_data))
    similarity_score = np.zeros(len(observed_data))
    hist_values = np.zeros(
        (1, int(len(observed_data[0]) / section_num),
         int(len(observed_data[0][0]))))   

    for sample in observed_data:   
        t1 += 1
        t2 = 0
        hist_data = np.zeros((1, int(len(sample[0]))))   

        for hour_data in sample[int(len(sample) / section_num) * (section_num - 1) + int( len(sample) % section_num):]:   
            t2 += 1
            t3 = 0
            best_score = 0   
            best_hist_for_hour = np.zeros(int(len(sample[0])))
            dist_sum = 0
            score_sum = 0
            for hist_temp in sample[:int(len(sample) / section_num) * (section_num - 1) + int( len(sample) % section_num)]:   
                t3 += 1
                dist_temp = 0
                score_temp = get_similarity_score(hour_data, hist_temp, dist_strategy)
                if best_score <= score_temp:
                    best_hist_for_hour = hist_temp
                    best_score = score_temp
                    dist_temp = int(len(sample) / section_num) * (section_num - 1) - t3 + 1
            dist_sum += dist_temp
            score_sum += best_score
            if t2 == 1:
                hist_data[0] = best_hist_for_hour
            else:
                hist_data = np.r_[hist_data, [best_hist_for_hour]]
        dist[t1 - 1] = dist_sum / (int(len(sample) / section_num) ** 2) + 1   
        similarity_score[t1 - 1] = score_sum / int(len(sample) / section_num)
        if t1 == 1:
            hist_values[0] = hist_data
        else:
            hist_values = np.r_[hist_values, [hist_data]]

    hist_masks = np.array(hist_values, dtype=bool)   
    hist_masks = hist_masks.astype("int64")

    return hist_values, hist_masks, dist, similarity_score


def matching_hist_entire(observed_data, history_data, dist_strategy, matching_times):   
    matched_hist = np.zeros((len(observed_data), int(matching_times), len(observed_data[0]), len(observed_data[0][0])))
    matched_dist = np.zeros((len(observed_data), int(matching_times)))
    matched_similarity_score = np.zeros((len(observed_data), int(matching_times)))

    for i in range(len(observed_data)):   
        score_list = np.zeros(len(history_data[i]) - len(observed_data[i]) + 1)
        target_values = observed_data[i].flatten()

        for j in range(len(history_data[i]) - len(observed_data[i]) + 1):   
            temp_values = history_data[i][j:j + len(observed_data[i])].flatten()
            score_list[j] = get_similarity_score(target_values, temp_values, dist_strategy)

         
        top_k_score_index = score_list.argsort()[:int(matching_times)]   
        top_k_score_sum = np.sum(score_list[top_k_score_index])   
        matched_dist[i] = 1 - (len(history_data[i]) - top_k_score_index - 1) / (len(history_data[i]))   
        for k in range(int(matching_times)):
            hist_k_value = history_data[i][top_k_score_index[k]:top_k_score_index[k] + len(observed_data[i])]
             
            matched_similarity_score[i][k] = 1 - (score_list[top_k_score_index[k]] / top_k_score_sum)   
            matched_hist[i][k] = hist_k_value

    return matched_hist, matched_dist, matched_similarity_score


def matching_hist(observed_data, history_data, dist_strategy, matching_times, match_strategy):   
    matched_hist = np.zeros((len(observed_data), int(matching_times), len(observed_data[0]), len(observed_data[0][0])))
    matched_dist = np.zeros((len(observed_data), int(matching_times)))
    matched_similarity_score = np.zeros((len(observed_data), int(matching_times)))

    for i in range(len(observed_data)):   
        score_list = np.zeros(int(matching_times))
        target_values = observed_data[i].flatten()
        hist_flag = []

         
        if match_strategy == "close":
            hist_flag = list(range(len(history_data[i]) + 1 - int(matching_times) - len(observed_data[i]), len(history_data[i]) + 1 - len(observed_data[i])))
         
        else:
            while len(hist_flag) < int(matching_times):
                 
                num_flag = random.randint(0, int(len(history_data[i]) - len(observed_data[i])))
                 
                if num_flag not in hist_flag:
                    hist_flag.append(num_flag)

        for j in range(int(matching_times)):
            temp_values = history_data[i][hist_flag[j]:hist_flag[j] + len(observed_data[i])]
            matched_hist[i][j] = temp_values
            matched_dist[i][j] = 1 - (len(history_data[i]) - hist_flag[j] - 1) / (len(history_data[i]))
            score_list[j] = get_similarity_score(target_values, temp_values.flatten(), dist_strategy)
            matched_similarity_score[i][j] = 1 - (score_list[j] / np.sum(score_list))

    return matched_hist, matched_dist, matched_similarity_score


def get_matched_values(observed_values, history_data, match_strategy, dist_strategy, matching_times):
    matched_hist = []
    matched_dist = []
    matched_similarity_score = []

    for i in range(int(matching_times)):
        if len(history_data[0]) < len(observed_values[0]):
            print("The length of history_data is shorter than observed_masks,so there is no more history information!")
            matching_times = i
            break
        if match_strategy == "segment":
            matched_hist, matched_dist, matched_similarity_score = matching_hist_segment(observed_values, history_data, dist_strategy, matching_times)
        elif match_strategy == "entire":
            matched_hist, matched_dist, matched_similarity_score = matching_hist_entire(observed_values, history_data,dist_strategy, matching_times)
        else:
            matched_hist, matched_dist, matched_similarity_score = matching_hist(observed_values, history_data, dist_strategy, matching_times, match_strategy)

    return matched_hist, matched_dist, matched_similarity_score, matching_times


def get_fusion_values(observed_values, matched_hist, matched_dist, matched_similarity_score, matching_times, hist_proportion):
    fusion_values = np.zeros(observed_values.shape)
    for j in range(len(observed_values)):
        lamda = (hist_proportion * matched_dist[j] + (1 - hist_proportion) * matched_similarity_score[j]) / (
                    hist_proportion * matched_dist.sum(axis=1)[j] + (1 - hist_proportion) *
                    matched_similarity_score.sum(axis=1)[j])
        for k in range(int(matching_times)):
            fusion_values[j] += lamda[k] * matched_hist[j][
                                           k * len(observed_values[j]):k * len(observed_values[j]) + len(
                                               observed_values[j])]

    return fusion_values


def get_condition_data(observed_values, hist_values):
    observed_masks = np.array(observed_values, dtype=bool).astype("int64")
    hist_masks = np.array(hist_values, dtype=bool).astype("int64")
    cond_values = np.zeros(observed_values.shape)

    for i in range(len(observed_values)):   
        hist_only = np.maximum(hist_masks[i] - observed_masks[i], 0)   
        cond_values[i] = hist_only * hist_values[i] + observed_masks[i] * observed_values[i]
    cond_masks = np.array(cond_values, dtype=bool).astype("int64")
    return cond_values, cond_masks


def get_missed_data(observed_values, missing_ratio, missing_type="random"):
    origin_masks = np.array(observed_values, dtype=bool).astype("int64")     
    observed_masks = origin_masks.copy()   
     
    for i in range(len(observed_values)):   
         
        obs_indices = np.where(origin_masks[i])[0].tolist()  
        missing_num = int(len(obs_indices) * missing_ratio)  
        if missing_num < 1:
            missing_num = 1
        elif missing_num > len(obs_indices) - 4:     
            missing_num = len(obs_indices) - 4
        observed_masks[i] = get_missed_mask(origin_masks[i], missing_num, missing_type)  

    observed_values = observed_values * observed_masks
    print("结束样本检查")

    return observed_values, observed_masks


def get_missed_mask(target_masks, missing_num, missing_type="random"):
     
    masks = target_masks.reshape(-1).copy().astype("int64")   
    masks_shape = masks.reshape(target_masks.shape)   

    obs_indices = np.where(masks)[0].tolist()   
    if missing_type == "random":
        miss_indices = np.random.choice(obs_indices, int(missing_num), replace=False).astype(
            "int64")   
    elif missing_type == "blackout":   
        miss_indices = obs_indices[int(len(obs_indices) - int(missing_num)):]

    masks[miss_indices] = False  
    missed_masks = masks.reshape(masks_shape.shape)   

    return missed_masks


class Dataset(Dataset):
    def __init__(self, current_day,num,eval_length=24, section_num=8, use_index_list=None, missing_ratio=0.1, seed=0,
                 use_hist_cond=False,
                 dist_strategy="L2", match_strategy="entire", matching_times=5, missing_type="random"):
        self.eval_length = eval_length
        self.section_num = section_num
        np.random.seed(
            seed)   
        self.observed_values = np.zeros((725,24,7))
        self.observed_masks = []
        self.origin_values = []
        self.origin_masks = []
        self.matched_values = []
        self.matched_dist = []
        self.matched_similarity_score = []
        self.missing_type = missing_type
        self.current_day = current_day
        self.num = num

        path = ("./data/ETTh1/ETTh1_725.pk")
         
        if os.path.isfile(path) == False:
            observed_origin = pd.read_csv("../matrix_filling_pq/data/ETTh1/ETTh1.csv", index_col="date", parse_dates=True).values[:, :]
            
            for i in range(725):   
                self.observed_values[i] = observed_origin[i * 24:(i + 1) * 24, :]

             
            np.nan_to_num(self.observed_values, copy=False)
             
            self.observed_masks = np.array(self.observed_values, dtype=bool).astype("int64")

             
             
             
            tmp_values = self.observed_values.reshape(-1, 7)
            tmp_masks = self.observed_masks.reshape(-1, 7)
            mean = np.zeros(7)
            std = np.zeros(7)
            for k in range(7):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()   
            self.observed_values = ((self.observed_values - mean) / std * self.observed_masks)

             
            np.nan_to_num(self.observed_values, copy=False, posinf=None, neginf=None)
            
            
             
            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks], f)
         
        else:   
            with open(path, "rb") as f:   
                self.observed_values, self.observed_masks = pickle.load(f)

    
        
         
        self.current_values = self.observed_values[self.current_day:self.current_day+self.num,:,:]   
        self.current_masks = self.observed_masks[self.current_day:current_day+self.num:,:,:]  


         
        self.origin_values = self.current_values   
        self.origin_masks = np.array(self.origin_values, dtype=bool).astype("int64")
         
         
        self.current_values, self.current_masks = get_missed_data(self.current_values, missing_ratio, missing_type)   
        
        
         
         
         

         
         

         
         
         
         
         
        self.matched_values = self.current_values
        self.matched_dist = np.zeros((len(self.current_values), int(matching_times)))
        self.matched_similarity_score = np.zeros((len(self.current_values), int(matching_times)))
        self.matching_times = 0
         
        if use_index_list is None:   
            self.use_index_list = np.arange(len(self.current_values))
        else:
            self.use_index_list = use_index_list   
        

    def __getitem__(self, org_index):
        if self.current_values.shape[0]>=2:
            self.use_index_list=torch.arange(0, self.current_values.shape[0])
         
         
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.current_values[index],
            "observed_mask": self.current_masks[index],
            "origin_data": self.origin_values[index],
            "origin_mask": self.origin_masks[index],
            "matched_data": self.matched_values[index],
            "timepoints": np.arange(self.eval_length),   
            "matched_dist": self.matched_dist[index],
            "matched_similarity_score": self.matched_similarity_score[index],
            "matching_times": self.matching_times,
            "index": index,
            "neighbor": self.current_values
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    def _values(self):
        return self.current_values

class neighbor_Dataset(Dataset):
    def __init__(self,eval_length=24, section_num=8, use_index_list=None, missing_ratio=0.1, seed=0,
                 use_hist_cond=False,
                 dist_strategy="L2", match_strategy="entire", matching_times=5, missing_type="random",neighbor_day=None,num=None):
        self.eval_length = eval_length
        self.section_num = section_num
        np.random.seed(
            seed)   
        self.observed_values = np.zeros((725,24,7))
        self.observed_masks = []
        self.origin_values = []
        self.origin_masks = []
        self.matched_values = []
        self.matched_dist = []
        self.matched_similarity_score = []
        self.missing_type = missing_type
        self.neighbor_day = neighbor_day
        self.num = num

        path = ("./data/ETTh1/ETTh1_725.pk")
         
        if os.path.isfile(path) == False:
            observed_origin = pd.read_csv("", index_col="date", parse_dates=True).values[:, :]
             
            for i in range(725):   
                self.observed_values[i] = observed_origin[i * 24:(i + 1) * 24, :]

             
            np.nan_to_num(self.observed_values, copy=False)
             
            self.observed_masks = np.array(self.observed_values, dtype=bool).astype("int64")

             
             
             
            tmp_values = self.observed_values.reshape(-1, 7)
            tmp_masks = self.observed_masks.reshape(-1, 7)
            mean = np.zeros(7)
            std = np.zeros(7)
            for k in range(7):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()   
            self.observed_values = ((self.observed_values - mean) / std * self.observed_masks)

             
            np.nan_to_num(self.observed_values, copy=False)
            np.inf_to_num(self.observed_values, copy=False)
            
             
            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks], f)
         
        else:   
            with open(path, "rb") as f:   
                self.observed_values, self.observed_masks = pickle.load(f)

        
         
        self.current_values = self.observed_values[neighbor_day:neighbor_day+num,:,:]   
        self.current_masks = self.observed_masks[neighbor_day:neighbor_day+num:,:,:]  
         
        history_values = self.observed_values[0:neighbor_day,:,:]   

         
        

         
        self.origin_values = self.current_values   
        self.origin_masks = np.array(self.origin_values, dtype=bool).astype("int64")
         
         
         
        
        
         
         
         

         
         

         
         
         
         
         
        self.matched_values = self.current_values
        self.matched_dist = np.zeros((len(self.current_values), int(matching_times)))
        self.matched_similarity_score = np.zeros((len(self.current_values), int(matching_times)))
        self.matching_times = 0
         
        if use_index_list is None:   
            self.use_index_list = np.arange(len(self.current_values))
        else:
            self.use_index_list = use_index_list   
        

    def __getitem__(self, org_index):
        if self.current_values.shape[0]>=2:
            self.use_index_list=torch.arange(0, self.current_values.shape[0])
         
        index = self.use_index_list[org_index]
         
        s = {
            "observed_data": self.current_values[index],
            "observed_mask": self.current_masks[index],
            "origin_data": self.origin_values[index],
            "origin_mask": self.origin_masks[index],
            "matched_data": self.matched_values[index],
            "timepoints": np.arange(self.eval_length),   
            "matched_dist": self.matched_dist[index],
            "matched_similarity_score": self.matched_similarity_score[index],
            "matching_times": self.matching_times,
            "index": index,
            
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    



def get_dataloader(current_day,num,seed=1, nfold=None, batch_size=16, missing_ratio=0.1, use_hist_cond=False, dist_strategy="L2",
                   match_strategy="entire", matching_times=5, missing_type="random"):
     
     
     
     

     
     

     
     

     
    
    
     
     
     
     
     

    np.random.seed(seed)   
     
     
     
     

     
    train_dataset = Dataset(current_day=current_day,num=num,missing_ratio=missing_ratio, seed=seed,
                                   use_hist_cond=False, dist_strategy=dist_strategy,
                                   match_strategy=match_strategy, matching_times=matching_times, missing_type=missing_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
     
     
     
     
    test_dataset = Dataset(current_day=current_day,num=num,missing_ratio=missing_ratio, seed=seed,
                                  use_hist_cond=False, dist_strategy=dist_strategy,
                                  match_strategy=match_strategy, matching_times=matching_times, missing_type=missing_type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader,  test_loader ,train_dataset._values()


def get_dataloader_add_history(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, use_hist_cond=False, dist_strategy="L2",
                   match_strategy="entire", matching_times=5, missing_type="random",neighbor_day=None,num = None):
     
     
     
     

   

    np.random.seed(seed)   
   
     
     
     
     
     

    neighbor_dataset =neighbor_Dataset(missing_ratio=missing_ratio, seed=seed,
                                   use_hist_cond=False, dist_strategy=dist_strategy,
                                   match_strategy=match_strategy, matching_times=matching_times, missing_type=missing_type,neighbor_day=neighbor_day,num=num)
    neighbor_loader = DataLoader(neighbor_dataset, batch_size=batch_size, shuffle=1)

     
     
     
     
    return neighbor_loader
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def random_missing_data(X, seed_num,i,sample_rate=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]):
    np.random.seed(seed_num)    
    train_indices = []
    test_indices = []
    
    total_elements = X.numel()
      

    all_indices = np.arange(total_elements)

    while True:
        
          
        num_missing_10 = int(total_elements * sample_rate[0])
        missing_indices_10 = np.random.choice(total_elements, num_missing_10, replace=False)
        train_indices.append(missing_indices_10)
        remaining_indices_10 = np.setdiff1d(all_indices, missing_indices_10)
        test_indices.append(np.array(list(remaining_indices_10)))
        if i==1 :
            break
        
          
        num_missing_20 = int(total_elements * sample_rate[1])   
        combined_missing_indices = set(missing_indices_10)    
        additional_missing_needed = num_missing_20 - num_missing_10    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_20 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_20)
        remaining_indices_20 = np.setdiff1d(all_indices, missing_indices_20)
        test_indices.append(np.array(list(remaining_indices_20)))
        if i==2 :
            break
        
          
        num_missing_30 = int(total_elements * sample_rate[2])   
        combined_missing_indices = set(missing_indices_20)    
        additional_missing_needed = num_missing_30 - num_missing_20    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_30 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_30)
        remaining_indices_30 = np.setdiff1d(all_indices, missing_indices_30)
        test_indices.append(np.array(list(remaining_indices_30)))
        if i==3 :
            break
    
          
        num_missing_40 = int(total_elements * sample_rate[3])   
        combined_missing_indices = set(missing_indices_30)    
        additional_missing_needed = num_missing_40 - num_missing_30    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_40 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_40)
        remaining_indices_40 = np.setdiff1d(all_indices, missing_indices_40)
        test_indices.append(np.array(list(remaining_indices_40)))
        if i==4 :
            break
    
          
        num_missing_50 = int(total_elements * sample_rate[4])   
        combined_missing_indices = set(missing_indices_40)    
        additional_missing_needed = num_missing_50 - num_missing_40    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_50 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_50)
        remaining_indices_50 = np.setdiff1d(all_indices, missing_indices_50)
        test_indices.append(np.array(list(remaining_indices_50)))
        if i==5 :
            break
    
          
        num_missing_60 = int(total_elements * sample_rate[5])   
        combined_missing_indices = set(missing_indices_50)    
        additional_missing_needed = num_missing_60 - num_missing_50    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_60 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_60)
        remaining_indices_60 = np.setdiff1d(all_indices, missing_indices_60)
        test_indices.append(np.array(list(remaining_indices_60)))
        if i==6 :
            break
    
          
        num_missing_70 = int(total_elements * sample_rate[6])   
        combined_missing_indices = set(missing_indices_60)    
        additional_missing_needed = num_missing_70 - num_missing_60    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_70 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_70)
        remaining_indices_70 = np.setdiff1d(all_indices, missing_indices_70)
        test_indices.append(np.array(list(remaining_indices_70)))
        if i==7 :
            break
    
          
        num_missing_80 = int(total_elements * sample_rate[7])   
        combined_missing_indices = set(missing_indices_70)    
        additional_missing_needed = num_missing_80 - num_missing_70    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_80 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_80)
        remaining_indices_80 = np.setdiff1d(all_indices, missing_indices_80)
        test_indices.append(np.array(list(remaining_indices_80)))
        if i==8 :
            break
    

          
        num_missing_90 = int(total_elements * sample_rate[8])   
        combined_missing_indices = set(missing_indices_80)    
        additional_missing_needed = num_missing_90 - num_missing_80    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_90 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_90)
        remaining_indices_90 = np.setdiff1d(all_indices, missing_indices_90)
        test_indices.append(np.array(list(remaining_indices_90)))
        if i==9:
            break

          
        num_missing_100 = int(total_elements * sample_rate[9])  
        combined_missing_indices = set(missing_indices_90)    
        additional_missing_needed = num_missing_100 - num_missing_90    
          
        remaining_indices = set(range(total_elements)) - combined_missing_indices
        additional_missing_indices = np.random.choice(list(remaining_indices), additional_missing_needed, replace=False)
          
        combined_missing_indices.update(additional_missing_indices)
        missing_indices_100 = np.array(list(combined_missing_indices))
        train_indices.append(missing_indices_100)
        remaining_indices_100 = np.setdiff1d(all_indices, missing_indices_100)
        test_indices.append(np.array(list(remaining_indices_100)))
        if i==10:
            break
    return train_indices[i-1],test_indices[i-1]

def whole_missing_data(X, seed_num, i, sample_rate=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]):
    np.random.seed(seed_num)    
    total_elements = X.numel()    
    
    max_rate = max(sample_rate)    
    max_num_missing = int(total_elements * max_rate)    
    
      
    s_max = total_elements - max_num_missing
    if s_max < 0:
        raise ValueError("数据量不足以支持最大缺失比例。")
    
      
    s = np.random.randint(0, s_max + 1)
    print(s)
      
    rate = sample_rate[i-1]
    num_missing = int(total_elements * rate)
    
      
    train_indices = np.arange(s, s + num_missing)
    
      
    test_part1 = np.arange(0, s)
    test_part2 = np.arange(s + num_missing, total_elements)
    test_indices = np.concatenate((test_part1, test_part2))
    
    return train_indices, test_indices



import torch

  
X = torch.randn(100, 100)

  
train_idx, test_idx = random_missing_data(X, seed_num=42, i=10)

  
train_idx_9, _ = random_missing_data(X, seed_num=42, i=9)
train_idx_8, _ = random_missing_data(X, seed_num=42, i=8)
print(np.all(np.isin(train_idx_8, train_idx_9)))    













  
  
  
  
    
  
  

  

  
        
  
  
  
  
  
  
  
  
        
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
        
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  




  
  
  
  
  


def apply_missing_data(X, missing_indices):
    X_missing = X.clone()
    X_missing_flat = X_missing.ravel()    
    for indices in missing_indices:
        X_missing_flat[indices] = 0.0    
    
    X_missing = X_missing_flat.reshape(X.shape)    
    
    return X_missing


"""

  
X = np.random.rand(3,4)

  
train_indices,test_indices= random_missing_data(X,i=1,seed_num=2)
print("train_indices=",train_indices)
print("test_indices=",test_indices)

  
train_list = apply_missing_data(X, train_indices)
train_array = np.array(train_list)
trian_tenor = torch.from_numpy(train_array).to(device)
test_list = apply_missing_data(X, test_indices)
test_array = np.array(test_list)
test_tenor = torch.from_numpy(test_array).to(device)
print("trian_tenor=",trian_tenor)
print("test_tenor=",test_tenor)

"""

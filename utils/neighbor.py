import numpy as np
import nanopq
import torch
import torch.nn.functional as F

def find_nearest_neighbors(matrix_new,time_sample_matrix,time_sample_matrix_fill,neighbor_num):
      
    matrix_new = matrix_new.astype(np.float32)
    matrix_new[np.isinf(matrix_new)] = 0.0
    matrix_new[np.isnan(matrix_new)] = 0.0
      
      
    M = time_sample_matrix_fill.shape[2]
      
    train_input = time_sample_matrix_fill.transpose((0,2,1,3)).reshape((int(time_sample_matrix_fill.shape[0]),-1))
      
      
    pq = nanopq.PQ(M,Ks=8)    
      
    print('...Start training codewords...')
    pq.fit(train_input)
      
    print('...Start Quantization...')
    matrix_past_train_code = pq.encode(train_input)
      
      
    query = matrix_new.transpose((1,0,2)).reshape((-1))
    DistanceTable = pq.dtable(query)
    dists = DistanceTable.adist(matrix_past_train_code)
    dists = torch.Tensor(dists)
      
    num_neighbor = neighbor_num
      
    min_values, topk_idxs = torch.topk(dists, num_neighbor, dim=-1, largest=False, sorted=True)
    
    print("min_values shape= ", min_values.shape)
    print("topk_idxs shape= ", topk_idxs.shape)
    print("time_sample_matrix shape= ", time_sample_matrix.shape)
      
      
    distance_sum = torch.sum(min_values)
      
    weight_neighbor = min_values / distance_sum
      
    weight_neighbor = F.normalize(weight_neighbor, p=1, dim=0)
      
    neighbor_matrix = time_sample_matrix[topk_idxs,:,:,:]
    if topk_idxs.numel() == 1:
        neighbor_matrix = np.expand_dims(neighbor_matrix, axis=0)


    return topk_idxs,neighbor_matrix,weight_neighbor
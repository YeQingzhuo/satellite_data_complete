import math
import numpy as np
import torch
import torch.nn as nn
from model.diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device   
        self.target_dim = target_dim   

        self.emb_time_dim = config["model"]["timeemb"]   
        self.emb_feature_dim = config["model"]["featureemb"]   
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_hist_condition = False

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim   
        if self.is_unconditional is False:
            self.emb_total_dim += 1   
        self.embed_layer = nn.Embedding(   
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]   
        config_diff["side_dim"] = self.emb_total_dim   

        input_dim = 1 if self.is_unconditional is True else 2   
        self.diffmodel = diff_CSDI(config_diff, input_dim)   

         
        self.num_steps = config_diff["num_steps"]   
        if config_diff["schedule"] == "quad":   
            self.beta = np.linspace(   
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta   
        self.alpha = np.cumprod(self.alpha_hat)   
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(
            1)   

    def time_embedding(self, pos, d_model=128):   
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)   
        position = pos.unsqueeze(2)   
        div_term = 1 / torch.pow(   
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(
            position * div_term)   
        pe[:, :, 1::2] = torch.cos(position * div_term)   
        return pe

    def get_randmask(self, observed_mask, missing_type):   
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask   
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)   
        observed_reverse_mask = observed_mask.permute(0, 2, 1).clone()   
        observed_for_mask = observed_reverse_mask.reshape(len(observed_mask), -1)   
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()   
            num_masked = math.floor(num_observed * sample_ratio)   

            if missing_type == "blackout" and self.use_hist_condition is True:
                ones_indices = (observed_for_mask[i] == 1).nonzero(as_tuple=True)
                indices_to_change = ones_indices[0][-num_masked:]    
                observed_for_mask[i][indices_to_change] = 0   
            else:
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1   
        if missing_type == "blackout" and self.use_hist_condition is True:
            rand_mask = observed_for_mask.reshape(observed_reverse_mask.shape).float()
            rand_mask = rand_mask.permute(0, 2, 1)   
        else:
            rand_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()   
        return rand_mask   

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):   
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)   
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)   
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))   
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)   

        side_info = torch.cat([time_embed, feature_embed], dim=-1)   
        side_info = side_info.permute(0, 3, 2, 1)   

        if self.is_unconditional is False:   
             
            side_mask = cond_mask.unsqueeze(1)   
            side_info = torch.cat([side_info, side_mask], dim=1)   

        return side_info

    def get_fusion_values(self, observed_values, matched_hist, matched_dist, matched_similarity_score, matching_times, hist_proportion):
        fusion_values = np.zeros(observed_values.shape)

        values = fusion_values.reshape(fusion_values.shape[0], -1).copy()   
         

        matched_hist = matched_hist.reshape(matched_hist.shape[0], matched_hist.shape[1], -1)
        observed_values = observed_values.reshape(observed_values.shape[0], -1)

        for i in range(len(observed_values)):
            lamda = (hist_proportion * matched_dist[i] + (1 - hist_proportion) * matched_similarity_score[i]) / (hist_proportion * matched_dist.sum(axis=1)[i] + (1 - hist_proportion) * matched_similarity_score.sum(axis=1)[i])
            matched_hist_mask = np.array(matched_hist[i], dtype=bool).reshape(matched_hist.shape[1], -1).copy().astype("int64")
            lamda_denominator = np.zeros(len(observed_values[i]))

            for j in range(len(values[i])):
                for k in range(int(matching_times)):
                    if matched_hist_mask[k][j] == 1:
                        lamda_denominator[j] += lamda[k]

            for m in range(len(values[i])):
                if lamda_denominator[m] > 0:
                    for n in range(int(matching_times)):
                        values[i][m] += matched_hist[i][n][m]*(lamda[n]/lamda_denominator[m])

        fusion_values = values.reshape(fusion_values.shape)   
        return fusion_values

    def get_fusion_values_average(self, observed_values, matched_hist, matched_dist, matched_similarity_score, matching_times, hist_proportion):
        fusion_values = np.zeros(observed_values.shape)
        lamda = 1/int(matching_times)

        for j in range(len(observed_values)):
            matched_hist_mask = np.array(matched_hist[j], dtype=bool).astype("int64")
            fusion_mask = matched_hist_mask.sum(axis=0)

            for k in range(int(matching_times)):
                hist_mask_k_only = np.maximum(matched_hist_mask[k] * 2 - fusion_mask, 0)   
                hist_mask_k_union = matched_hist_mask[k] - hist_mask_k_only
                fusion_values[j] += hist_mask_k_union * lamda * matched_hist[j][k] + hist_mask_k_only * matched_hist[j][k]

        return fusion_values

    def get_condition_data(self, observed_values, fusion_values):
        observed_masks = np.array(observed_values, dtype=bool).astype("int64")
        hist_masks = np.array(fusion_values, dtype=bool).astype("int64")
        cond_values = np.zeros(observed_values.shape)

        for i in range(len(observed_values)):   
            hist_only = np.maximum(hist_masks[i] - observed_masks[i], 0)   
            cond_values[i] = hist_only * fusion_values[i] + observed_masks[i] * observed_values[i]
        cond_masks = np.array(cond_values, dtype=bool).astype("int64")
        return cond_values, cond_masks

    def calc_loss_valid(self, observed_data, observed_mask, cond_data, cond_mask, rand_mask, side_info, is_train):   
        loss_sum = 0
        for t in range(self.num_steps):   
            loss = self.calc_loss(
                observed_data, observed_mask, cond_data, cond_mask, rand_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, observed_mask, cond_data, cond_mask, rand_mask, side_info, is_train, set_t=-1):   
        B, K, L = observed_data.shape
        if is_train != 1:   
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:   
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]   
        noise = torch.randn_like(observed_data)   
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, cond_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)   
         
         
        target_mask = observed_mask - rand_mask   
        residual = (noise - predicted) * target_mask   
        num_eval = target_mask.sum()   
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        if loss != loss:   
            print("loss is NAN.")
             
             
             
        return loss

    def set_input_to_diffmodel(self, noisy_data, cond_data, cond_mask):
        if self.is_unconditional is True:
            total_input = noisy_data.unsqueeze(1)   
        else:
             
            cond_obs = (cond_mask * cond_data).unsqueeze(1)   

            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)   
            total_input = torch.cat([cond_obs, noisy_target], dim=1)   

        return total_input

    def impute(self, observed_data, observed_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
             
            if self.is_unconditional is True:   
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * observed_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):   
                if self.is_unconditional is True:   
                    diff_input = observed_mask * noisy_cond_history[t] + (1.0 - observed_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)   
                else:
                     
                    cond_obs = (observed_mask * observed_data).unsqueeze(1)

                    noisy_target = ((1 - observed_mask) * current_sample).unsqueeze(1)   
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)   
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))   

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)   

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, missing_type, is_train=1):   
        (
            observed_data,   
            observed_mask,   
            origin_data,   
            origin_mask,
            matched_data,   
            observed_tp,   
            for_pattern_mask,   
            _,   
            matched_dist,   
            matched_similarity_score,   
            matching_times,   
            index
        ) = self.process_data(batch)

        if is_train == 0:
            rand_mask = observed_mask
        else:
            rand_mask = self.get_randmask(observed_mask, missing_type)   

        if self.use_hist_condition is True:
            observed_values = observed_data.permute(0, 2, 1).cpu().numpy()   
            fusion_values = self.get_fusion_values(observed_values, matched_data, matched_dist, matched_similarity_score, matching_times, 0.3)   
            cond_values, cond_masks = self.get_condition_data(observed_values, fusion_values)
            cond_data = torch.tensor(cond_values).to(self.device).float().permute(0, 2, 1)   
            cond_mask = torch.tensor(cond_masks).to(self.device).float().permute(0, 2, 1)   
            cond_mask = torch.tensor(np.maximum(cond_mask.cpu().numpy() - (observed_mask.cpu().numpy() - rand_mask.cpu().numpy()), 0)).to(self.device).float()   
        else:
            cond_data = observed_data
            cond_mask = rand_mask

        side_info = self.get_side_info(observed_tp, rand_mask)   

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, observed_mask, cond_data, cond_mask, rand_mask, side_info, is_train), index   

    def evaluate(self, batch, hist_proportion, n_samples):   
        (
            observed_data,   
            observed_mask,   
            origin_data,
            origin_mask,
            matched_data,
            observed_tp,   
            _,
            cut_length,   
            matched_dist,
            matched_similarity_score,
            matching_times,
            index
        ) = self.process_data(batch)

        with torch.no_grad():
            target_mask = origin_mask - observed_mask
             
             
             
             
             
                 
                 
                 
                 
                 

            side_info = self.get_side_info(observed_tp, observed_mask)

            samples = self.impute(observed_data, observed_mask, side_info, n_samples)    

             
             
             
             
            samples_median = samples.median(dim=1)
             
            samples_final = observed_data*observed_mask + samples_median.values*(1-observed_mask)

            for i in range(len(cut_length)):   
                target_mask[i, ..., 0: cut_length[i].item()] = 0
        return samples, samples_final, origin_data, target_mask, origin_mask, observed_tp




class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):   
         
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        origin_data = batch["origin_data"].to(self.device).float()
        origin_mask = batch["origin_mask"].to(self.device).float()
        matched_data = batch["matched_data"].to(self.device).float().cpu().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()
        matched_dist = batch["matched_dist"].to(self.device).float().cpu().numpy()
        matched_similarity_score = batch["matched_similarity_score"].to(self.device).float().cpu().numpy()
        matching_times = batch["matching_times"].to(self.device).float().cpu().numpy()[0]
        index = batch["index"].to(self.device).float().cpu().numpy()

        print("observed_data_shape:", observed_data.shape)
        observed_data = observed_data.permute(0, 2, 1)  
        observed_mask = observed_mask.permute(0, 2, 1)  
        origin_data = origin_data.permute(0, 2, 1)
        origin_mask = origin_mask.permute(0, 2, 1)

        for_pattern_mask = observed_mask   
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)   

        return (
            observed_data,
            observed_mask,
            origin_data,
            origin_mask,
            matched_data,
            observed_tp,
            for_pattern_mask,
            cut_length,
            matched_dist,
            matched_similarity_score,
            matching_times,
            index
        )




class CSDI_ETTh1(CSDI_base):
    def __init__(self, config, device, target_dim=7):
        super(CSDI_ETTh1, self).__init__(target_dim, config, device)

    def process_data(self, batch):   
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        origin_data = batch["origin_data"].to(self.device).float()
        origin_mask = batch["origin_mask"].to(self.device).float()
        matched_data = batch["matched_data"].to(self.device).float().cpu().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()
        matched_dist = batch["matched_dist"].to(self.device).float().cpu().numpy()
        matched_similarity_score = batch["matched_similarity_score"].to(self.device).float().cpu().numpy()
        matching_times = batch["matching_times"].to(self.device).float().cpu().numpy()[0]
        index = batch["index"].to(self.device).float().cpu().numpy()

        observed_data = observed_data.permute(0, 2, 1)  
        observed_mask = observed_mask.permute(0, 2, 1)  
        origin_data = origin_data.permute(0, 2, 1)
        origin_mask = origin_mask.permute(0, 2, 1)

        for_pattern_mask = observed_mask   
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)   

        return (
            observed_data,
            observed_mask,
            origin_data,
            origin_mask,
            matched_data,
            observed_tp,
            for_pattern_mask,
            cut_length,
            matched_dist,
            matched_similarity_score,
            matching_times,
            index
        )





class CSDI_Satellite(CSDI_base):
    def __init__(self, config, device, target_dim=11):
        super(CSDI_Satellite, self).__init__(target_dim, config, device)

    def process_data(self, batch):   
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        origin_data = batch["origin_data"].to(self.device).float()
        origin_mask = batch["origin_mask"].to(self.device).float()
        matched_data = batch["matched_data"].to(self.device).float().cpu().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()
        matched_dist = batch["matched_dist"].to(self.device).float().cpu().numpy()
        matched_similarity_score = batch["matched_similarity_score"].to(self.device).float().cpu().numpy()
        matching_times = batch["matching_times"].to(self.device).float().cpu().numpy()[0]
        index = batch["index"].to(self.device).float().cpu().numpy()

        observed_data = observed_data.permute(0, 2, 1)  
        observed_mask = observed_mask.permute(0, 2, 1)  
        origin_data = origin_data.permute(0, 2, 1)
        origin_mask = origin_mask.permute(0, 2, 1)

        for_pattern_mask = observed_mask   
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)   

        return (
            observed_data,
            observed_mask,
            origin_data,
            origin_mask,
            matched_data,
            observed_tp,
            for_pattern_mask,
            cut_length,
            matched_dist,
            matched_similarity_score,
            matching_times,
            index
        )



class CSDI_sun(CSDI_base):
    def __init__(self, config, device, target_dim=12):
        super(CSDI_sun, self).__init__(target_dim, config, device)

    def process_data(self, batch):   
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        origin_data = batch["origin_data"].to(self.device).float()
        origin_mask = batch["origin_mask"].to(self.device).float()
        matched_data = batch["matched_data"].to(self.device).float().cpu().numpy()
        observed_tp = batch["timepoints"].to(self.device).float()
        matched_dist = batch["matched_dist"].to(self.device).float().cpu().numpy()
        matched_similarity_score = batch["matched_similarity_score"].to(self.device).float().cpu().numpy()
        matching_times = batch["matching_times"].to(self.device).float().cpu().numpy()[0]
        index = batch["index"].to(self.device).float().cpu().numpy()

        observed_data = observed_data.permute(0, 2, 1)  
        observed_mask = observed_mask.permute(0, 2, 1)  
        origin_data = origin_data.permute(0, 2, 1)
        origin_mask = origin_mask.permute(0, 2, 1)

        for_pattern_mask = observed_mask   
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)   

        return (
            observed_data,
            observed_mask,
            origin_data,
            origin_mask,
            matched_data,
            observed_tp,
            for_pattern_mask,
            cut_length,
            matched_dist,
            matched_similarity_score,
            matching_times,
            index
        )

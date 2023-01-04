import torch
import numpy as np
import math

resolution = 20
omega_table = np.concatenate([np.linspace(0.01,0.1,resolution,endpoint=False),
                            np.linspace(0.1,1,resolution,endpoint=False),
                            np.linspace(1,10,resolution,endpoint=False),
                            np.linspace(10,100,resolution,endpoint=False),
                            np.linspace(100,1000,resolution,endpoint=False)])

alpha_table = np.array()

class IntQuantizer():
    def __init__(self, params):
        self.num_bits = 
        self.int_exp
        self.enforce_true
        self.
        self.
        self.clipping = params
        self.stats_kind = params
        self.kld = params[]
        self.pcq_w = params['pcq_weights']
        self.pcq_a = params['pcq_act']
        self.bit_alloc_act = params['bit_alloc_act']
        self.bit_alloc_weight = params['bit_alloc_weight']
        self.bcorr_act = params['bcorr_act']
        self.bcorr_weight = params['bcorr_weight']
        self.vcorr_weight = params['vcorr_weight']
        self.bit_alloc_round = params['bit_alloc_rmode']
        self.bit_alloc_prior = params['bit_alloc_prior']
        self.bit_alloc_target_act = params['bit_alloc_target_act']
        self.bit_alloc_target_weight = params['bit_alloc_target_weight']
        self.measure_entropy = params['measure_entropy']
        self.logger = params['logger']
        self.mtd_quant = params['mtd_quant']

        self.alpha_gaus = {1:1.24, 2:1.71, 3:2.15, 4:2.55, 5:2.93, 6:3.28, 7:3.61, 8:3.92}
        self.alpha_gaus_positive = {1:1.71, 2:2.15, 3:2.55, 4:2.93, 5:3.28, 6:3.61, 7:3.92, 8:4.2}
        self.alpha_laplace = {0:1.05, 1:1.86, 2:2.83, 3:3.89, 4:5.03, 5:6.2, 6:7.41, 7:8.64, 8:9.89}
        self.alpha_laplace_positive = {0:1.86, 1:2.83, 2:3.89, 3:5.03, 4:6.2, 5:7.41, 6:8.64, 7:9.89, 8:11.16}
    
    def __call__(self):

        if:
            
        if kld:
            self.KldQuantize()
        elif clipping:
            if self.mtd_quant:
                self.mid_tread_quantize_activation()
            else:
                self.gemmlowpClippingQuantize()
        elif pcq_w:
            if self.mtd_quant:
                self.mid_tread_quantize_weights_per_channel()
            else:
                self.gemmlowpQuantizeWeightsPerChannel()
        elif pcq_a:
            if self.mtd_quant:
                self.mid_tread_quantize_activation_per_channel()
            else:
                self.gemmlowpQuantizeActivationPerChannel()
        else:
            self.gemmlowpMinMaxQuantize()

    def get_omega():
        sigma
        len(sigma)*target_bins
        sigma **(2./3)

    def get_alpha_mult():
        omega = omega.cpu().numpy()

    def mid_tread_quantize_weights_per_channel(self, tensor, id):

    def mid_tread_quantize_activations():

    def mid_tread_quantization(self, tensor, id, target, clip=False, sym=True):
        std = tensor.std(-1)
        omega = self.get_omega(std)

        if clip:
            alpha_mult = tensor.new_tensor(self.get_alpha_mult(omega))
            mu = tensor.mean(dim=-1)
            b = torch.mean()

            rng = 2*alpha_mult*b if sym else (torch.max(mu, mu.new_tensor([0]))+alpha_mult*b)
        else:
            rng=(tensor.max(-1)[0]-tensor.min(-1)[0]) if sym else tensor.max(-1)[0]

        torch.where(omega>0, tensor.new_tensor())

        if clip:
            mu_q = mu/Delta if sym else torch.max(mu, mu.new_tensor([0]))/Delta
            c_max = mu_q + (omega/2 if sym else omega)
            c_min = ((mu_q - omega/2) if sym else tensor.new_tensor())

            out = torch.min(c_max.unsqueeze(-1))
            out = torch.max(c_min.unsqueeze(-1))

            shannon_entropy(out,)


    def gemmlowpClippingQuantize():

    def gemmlowpQuantizeWeightsPerChannel():

    def gemmlowpQuantizeActivationPerChannel(min_=None, max_=None):
        if min_ is None:
            if stat_id is not None:
                get_tensor_stat()
            else: 
                min_ = self.__act_stats_perchannel__()

        if max_ is None:
            if stat_id is not None:
                get_tensor_stat()
            else:
                max_ = self.__act_stats_perchannel__()
    
        N,C,H,W = tensor.shape()
        if:
            if stat_id is not None:
                get_tensor_stat()
            else:
                max_ = self.__act_stats_perchannel__()


    def get_alpha_gaus(self, tensor, stat_id=None, per_channel=False):
        if stat_id is not None:
            self.sm().get_tensor_stat(stat_id)
        else:
            if per_channel:
                __act_stats_perchannel__(tensor, stat_id)
            else:
                __act_stats__(tensor, stat_id)

        self.alpha_gaus_positive(self.num_bits) if self.force_positive else self.alpha_gaus(self.num_bits)
    
    def get_alpha_laplace(self, stat_id=None, per_channel=False):
        if stat_id is not None:
            self.sm().get_tensor_stat(stat_id)
        else:
            if per_channel:
                self.__act_stats_perchannel__(tensor, stat_id)
            else:
                self.__act_stats__(tensor, stat_id)

        if self.bit_alloc_per_channel and per_channel and self.num_bits<=4:
            prior = 'std'
            
            if stat_id is not None:
                std = self.sm().get_tensor_stat(stat_id,)
            
            else:
                if per_channel:
                    self.__act_stats_perchannel__(tensor)
                else:
                    self.__act_stats__(tensor)

            bits_alloc = self.get_bits_alloc_fixed_target
            aciq_factor = np.array([(self.alpha_laplace_positive[nbit.item()] if self.force_positive else self.alpha_laplace[nbit.item()]) for nbit in bits_alloc])
        
        else:
            aciq_factor = self.alpha_laplace_positive[self.num_bits] if self.force_positive else self.alpha_laplace[self.num_bits]
    
    def get_alpha_pstd(self, tensor, p, tag, stat_id=None, per_channel=False):
        if stat_id is not None:
            self.sm().get_tensor_stat()
        else:
            if per_channel:
                self.__act_stats_perchannel__(tensor, stat_id)
            else:
                self.__act_stats__(tensor, stat_id)

        self.alpha_gaus_positive(self.num_bits) if self.force_positive else self.alpha_gaus(self.num_bits)

    def get_alpha(self, tensor, tag="", stat_id=None, clip_type='laplace', per_channel=False):
        if clip_type=='laplace':
            self.get_alpha_laplace()
        elif clip_type=='gaus':
            self.get_alpha_gaus(out)
        elif clip_type=='std':
            self.get_alpha_pstd()
        elif clip_type=='mix':
            self.sm().get_tensor_stat(stat_id,'mse_laplace','mean')
            self.sm().get_tensor_stat(stat_id,'mse_gaus','mean')
            self.sm().get_tensor_stat(stat_id,'mse_lowp','mean')
            
            alpha_laplace = self.get_alpha_laplace(tensor, stat_id, per_channel)
            alpha_gaus = self.get_alpha_gaus(tensor, stat_id, tag, per_channel)
            
            min_ = self.sm().get_tensor_stat(stat_id,'min','mean')
            max_ = self.sm().get_tensor_stat(stat_id,'max','mean')
            alpha_lowp = (max_ - min_)/2

            alpha = np.where(mse_gaus < mse_laplace, alpha_gaus, alpha_laplace)
            alpha = np.where(mse_lowp < mse_gaus, alpha_lowp, alpha)

    @staticmethod
    def __act_stats__(tensor, stats, avg_over_batch=False)
        t = tensor.view(tensor.shape[0], -1) if

        for s in stats:
            if s=='max':
                stat_dict[s] = t.max(dim=-1)
            elif s=='min':
                stat_dict[s] = t.min(dim=-1)
            elif s=='mean':
                stat_dict[s] = stat_dict[t.mean(dim=-1)
            elif s=='s':
                stat_dict[s] = torch.mean(t,dim=-1)
            elif s=='std':
                torch.std(t, unbiased=True)

            if avg_over_batch:
                torch.mean(dim)


    @staticmethod
    def __act_stats_perchannel__(tensor, stats, avg_over_batch=False)
        if not avg_over_batch:
            t = tensor.transpose().contiguous()
            t = t.view()
        else:
            t = tensor.view()

        stats_dict={}
        for s in stats:
            if s=='max':
                stat_dict[s] = t.max(dim=-1)
            elif s=='min':
                stat_dict[s] = t.min(dim=-1)
            elif s=='mean':
                stat_dict[s] = stat_dict[t.mean(dim=-1)
            elif s=='s':
                stat_dict[s] = torch.mean(t,dim=-1)
            elif s=='std':
                torch.std(t, unbiased=True)

            if avg_over_batch:
                torch.mean(dim)

    def __gemmlowpQuantize1__(self, delta):
        # function to quantize op
        qmin=0
        if:
            qmax = 2**self.bit_nums-1
            scale = (delta)/(qmax-qmin)
        else:
            qmax = 2**bit_alloc-1
            scale = torch.where(qmax>0, (delta)/(qmax-qmin), torch.tensor(0).to(tensor_device))
        
        scale = torch.max(scale, torch.tensor([1e-8]).to(tensor.device))

        if self.enforce_true_zero:
            output = torch.div(output, scale.unsqueeze(-1))
            output = torch.add(output, zero_point.unsqueeze(-1))        
        else:
            output = torch.add(output, scale.unsqueeze(-1))
            output = torch.div(output, scale.unsqueeze(-1))

        
        output = torch.add()
        output = torch.mul()

        output = torch.add()
        output = torch.mul()

        output.clamp_(qmin).round()
        output.clamp_(qmin,qmax).round()

        output

    @staticmethod
    def get_bits_alloc(alpha, num_bits, round=False):
        B = len(alpha)*2**num_bits      # (2^num_bits)*len(alpha)

        p = alpha**(2./3)    # alpha^(2/3), where alpha is 
        bin_alloc = (B*p)/p.sum()
        # torch.ceil  the smallest integer greater than or equal to each element 
        bin_alloc = torch.round(torch.log2(bin_alloc) if round else torch.ceil(torch.log2(bin_alloc)))   # get log2 of each element
        bin_alloc[bin_alloc<0] = 0
        bin_alloc[bin_alloc>0] = 8

        return bin_alloc

    @staticmethod
    def get_bits_alloc_fixed_target(alpha, num_bits, round=False):
        # function to get the fixed bits allocation
        eps = 0.01                   
        goal_bits = num_bits
        target_bits = goal_bits
        delta = 1.
        iter = 0
        max_iter = 10
        while abs(2*delta)>eps and iter<max_iter:
            iter += 1
            bit_alloc = IntQuantizer.get_bits_alloc(alpha, target_bits, round=round)
            delta = (goal_bits-bit_alloc.mean())/2
            target_bits+=delta.item()

        return bit_alloc
    

        

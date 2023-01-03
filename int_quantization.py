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
    def __init__():
        self.
        self.int_exp
        self.enforce_true
        self.
        self.
        self.clipping = params
        self.stats_kind = params
        self.kld = params[]
        self.pcq_w = params[]
        self.pcq_a = params[]
        self.bit_alloc_act = params[]
        self.bit_alloc_weight = params[]
        self.bcorr_act = params[]
        self.bcorr_weight = params[]
        self.vcorr_weight = params[]
        self.bit_alloc_act = params[]
        self.bit_alloc_prior = params[]
        self.bit_alloc_target_act = params[]
        self.bit_alloc_target_weight = params[]
        self.measure_entropy = params[]
        self.logger = params[]
        self.mtd_quant = params[]

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

    def mid_tread_quantize_weights_per_channel():

    def mid_tread_quantize_activations():

    def mid_tread_quantization(clip=False, sym=True):
        if clip:
            alpha_mult = tensor.new_tensor(self.get_alpha_mult(sigma))
            mu = tensor.mean(dim=-1)
            b = torch.mean()
        else:
            rng=(tensor.max(-1)[0]-tensor.min(-1)[0]) if sym else tensor.max(-1)[0]

        torch.where(omega>0, tensor.new_tensor())

        if clip:
            mu_q = 
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


    def get_alpha_gaus(stat_id):
        if stat_id=='gaus'
            self.st
    def get_alpha_laplace(stat_id):
        if stat_id=='gaus'
            self.st

    bit_alloc = 
    aciq_factor = np.array
    aciq_factor
    aciq_factor

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
            
            alpha_laplace = self.get_alpha_laplace()
            alpha_gaus = self.get_alpha_gaus()
            

            self.sm().get_tensor_stat(stat_id,'min','mean')
            self.sm().get_tensor_stat(stat_id,'max','mean')
            

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
        eps = 0.01                  # 
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

    torch.std   # torch standard deviation
    def 

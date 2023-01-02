omega_table = np.concatenate([np.linspace(0.01,0.1,resolution,endpoint),
                            np.linspace(0.1,1,resolution,endpoint),
                            np.linspace(1,10,resolution,endpoint),
                            np.linspace(10,100,resolution,endpoint),
                            np.linspace(100,1000,resolution,endpoint)])

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

    def get_alpha():
        self.get_alpha_gaus(out)
        self.get_alpha_laplace()
        self.get_alpha_pstd()

    def __act_stats_perchannel__(tensor, stats)
        t = tensor.transpose().contiguous()
        t = 

        for s in stats:
            if s=='max':
                t.max(dim=-1)
            elif s=='min':
                t.min(dim=-1)
            elif s=='mean':
                t.mean(dim=-1)

            elif:
                torch.mean()


    loss = 
    def __gemmlowpQuantize1__():
        # function to quantize op
        if:
            qmax = 2**self.bit_nums-1
        else:
            qmax = 2**bit_alloc-1
        
        elif:
            output = torch.div()
            output = torch.add()
        
        elif:
            output = torch.div()
            output = torch.add()

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import deque

class ParallelZOOptimizer:
    def __init__(self, model: nn.Module, lr=1e-3, eps=1e-3, weight_decay=0.0, device='cuda:0'):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.device = device
        
        # DDP Info
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Pair Grouping
        # Assuming world_size is even.
        # Pairs: (0,1), (2,3), ...
        self.num_pairs = self.world_size // 2
        self.pair_id = self.rank // 2
        self.role = self.rank % 2 # 0: Positive (+z), 1: Negative (-z)
        self.partner_rank = self.rank ^ 1
        
        # Create a process group for the pair to efficiently exchange loss
        self._setup_pair_group()
        
        self.projected_grad = 0.0
        self.current_step_seed = None
        self.last_step_seed = None
        self.seed_queue = deque(maxlen=2)
        
        # Module Discovery
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            # NanoGPT structure
            self.modules_list = [
                model.transformer.wte,      
                model.transformer.wpe,      
                *model.transformer.h,       
                model.transformer.ln_f,     
                model.lm_head               
            ]
        else:
            # Generic fallback (e.g. Qwen, generic HF models)
            self.modules_list = [model]

    def _setup_pair_group(self):
        # Create subgroups for each pair
        self.pair_group = None
        for i in range(0, self.world_size, 2):
            ranks = [i, i+1]
            group = dist.new_group(ranks=ranks)
            if self.rank in ranks:
                self.pair_group = group

    # ============================================================
    # Original distZO2 methods (for module-level perturbation)
    # ============================================================

    def step_start_init(self, seed):
        self.current_step_seed = seed
        self.seed_queue.append(seed)
        if len(self.seed_queue) == 2:
            self.last_step_seed = self.seed_queue[0]

    def _apply_op(self, module, seed, op_type, **kwargs):
        torch.cuda.manual_seed(seed)
        for name, param in module.named_parameters():
             if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                
                if op_type == "perturb":
                    eff_scaling = 1 if self.role == 0 else -1
                    param.data.add_(z * self.eps * eff_scaling)
                    
                elif op_type == "update":
                    no_decay = any(nd in name for nd in ["bias", "layer_norm", "layernorm", "ln", "norm"])
                    if no_decay:
                        param.data.sub_(self.lr * self.projected_grad * z)
                    else:
                        param.data.sub_(self.lr * (self.projected_grad * z + self.weight_decay * param.data))

    def parallel_perturb(self):
        for i, module in enumerate(self.modules_list):
            self._apply_op(module, seed=self.current_step_seed + i, op_type="perturb")

    def parallel_compute_grad(self, loss):
        loss_tensor = torch.tensor([loss.item()], device=self.device)
        output_list = [torch.zeros_like(loss_tensor) for _ in range(2)]
        dist.all_gather(output_list, loss_tensor, group=self.pair_group)
        
        l1 = output_list[0].item()
        l2 = output_list[1].item()
        local_grad = (l1 - l2) / (2 * self.eps)
        
        grad_tensor = torch.tensor([local_grad], device=self.device)
        dist.all_reduce(grad_tensor, op=dist.ReduceOp.AVG)
        self.projected_grad = grad_tensor.item()
        
        inverse_scaling = -1 if self.role == 0 else 1
        for i, module in enumerate(self.modules_list):
            self._apply_op_restore(module, seed=self.current_step_seed + i, scaling=inverse_scaling)

    def _apply_op_restore(self, module, seed, scaling):
        torch.cuda.manual_seed(seed)
        for name, param in module.named_parameters():
             if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                param.data.add_(z * self.eps * scaling)

    def zo_update(self):
        if self.projected_grad == 0.0 or self.last_step_seed is None: return
        for i, module in enumerate(self.modules_list):
            self._apply_op(module, seed=self.last_step_seed + i, op_type="update")

    # ============================================================
    # Extended methods for dp-aggzo integration
    # These work with named_parameters lists (parameter-level)
    # instead of modules_list (module-level), making them
    # compatible with dp-aggzo's training loop.
    # ============================================================

    def perturb_parameters_by_named(self, named_parameters, seed):
        """
        Perturb parameters based on role: +eps*z for role 0, -eps*z for role 1.
        Works with a list of (name, param) tuples (dp-aggzo style).
        """
        torch.manual_seed(seed)
        scaling = 1 if self.role == 0 else -1
        scalar = scaling * self.eps
        for name, param in named_parameters:
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scalar * z
                del z

    def restore_parameters_by_named(self, named_parameters, seed):
        """
        Restore parameters by undoing the perturbation (inverse of perturb_parameters_by_named).
        """
        torch.manual_seed(seed)
        inverse_scaling = -1 if self.role == 0 else 1
        scalar = inverse_scaling * self.eps
        for name, param in named_parameters:
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.data.size(),
                                device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scalar * z
                del z

    def exchange_loss(self, loss):
        """
        Exchange loss tensors within a GPU pair via all_gather.
        Supports both scalar and per-sample loss vectors (for DP).
        
        Returns:
            (loss_pos, loss_neg): loss from positive-perturbation rank and negative-perturbation rank.
        """
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        loss_tensor = loss.clone().to(self.device)
        output_list = [torch.zeros_like(loss_tensor) for _ in range(2)]
        dist.all_gather(output_list, loss_tensor, group=self.pair_group)
        return output_list[0], output_list[1]  # loss_pos, loss_neg

    def distribute_directions(self, K):
        """
        Distribute K random directions across GPU pairs.
        Each pair gets K // num_pairs directions; first K % num_pairs pairs get one extra.
        
        Returns:
            List of direction indices assigned to this pair.
        """
        P = self.num_pairs
        assert K >= P, f"Number of directions ({K}) must be >= number of GPU pairs ({P})"
        K_per_pair = K // P
        K_remainder = K % P

        if self.pair_id < K_remainder:
            my_start = self.pair_id * (K_per_pair + 1)
            my_count = K_per_pair + 1
        else:
            my_start = K_remainder * (K_per_pair + 1) + (self.pair_id - K_remainder) * K_per_pair
            my_count = K_per_pair

        return list(range(my_start, my_start + my_count))

    def gather_grads_across_pairs(self, local_grads, my_directions, total_K):
        """
        Gather gradient estimates from all GPU pairs using all_reduce.
        Only even-ranked GPUs (role 0) contribute to avoid double-counting.
        
        Args:
            local_grads: list of gradient tensors computed by this pair
            my_directions: list of direction indices assigned to this pair
            total_K: total number of directions
            
        Returns:
            List of total_K gradient tensors (one per direction).
        """
        grad_dim = local_grads[0].shape[0]
        all_grads_buffer = torch.zeros(total_K, grad_dim, device=self.device)
        if self.role == 0:  # Only even ranks contribute
            for i, d_idx in enumerate(my_directions):
                all_grads_buffer[d_idx] = local_grads[i]
        dist.all_reduce(all_grads_buffer, op=dist.ReduceOp.SUM)
        return [all_grads_buffer[j] for j in range(total_K)]

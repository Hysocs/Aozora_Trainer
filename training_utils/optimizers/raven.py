import math

import torch
from torch.optim import Optimizer


class RavenAdamW(Optimizer):
    """
    AdamW-style optimizer that keeps update math in FP32 while storing momentum
    state on the CPU in a configurable dtype.

    Raven is built for large training runs where optimizer state movement is a
    noticeable cost. It reuses one FP32 scratch buffer on the training device,
    copies the stored first and second moments into that buffer for the update,
    applies the AdamW step there, then writes the updated state back to CPU
    storage.

    Compared with a plain FP32 CPU-state AdamW variant, Raven can reduce CPU
    RAM usage and CPU-to-device transfer volume when `momentum_dtype` is set to
    `torch.bfloat16` or `torch.float16`, while preserving FP32 math for the
    actual parameter update. `torch.bfloat16` is the intended default because
    it keeps FP32-like exponent range and is safer for the second-moment state.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 0.06,
        eps: float = 1e-8,
        debias_strength: float = 1.0,
        momentum_dtype: torch.dtype = torch.bfloat16,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        valid_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        if momentum_dtype not in valid_dtypes:
            raise ValueError(
                f"momentum_dtype must be one of {valid_dtypes}, got {momentum_dtype}"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            debias_strength=debias_strength,
            momentum_dtype=momentum_dtype,
        )
        super(RavenAdamW, self).__init__(params, defaults)

        self.max_numel = 0
        self.param_device = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if self.param_device is None:
                        self.param_device = p.device
                    self.max_numel = max(self.max_numel, p.numel())

        if self.max_numel > 0:
            self._scratch_buffer = torch.zeros(
                3 * self.max_numel,
                device=self.param_device,
                dtype=torch.float32,
            )
        else:
            self._scratch_buffer = None

        self._momentum_dtype = momentum_dtype

    def _get_scratch_buffers(self, p):
        numel = p.numel()
        return (
            self._scratch_buffer[:numel].view_as(p),
            self._scratch_buffer[numel:2 * numel].view_as(p),
            self._scratch_buffer[2 * numel:3 * numel].view_as(p),
        )

    def _new_cpu_state_tensor(self, p, dtype):
        return torch.zeros_like(p, device="cpu", dtype=dtype)

    def _restore_cpu_state_tensor(self, tensor):
        return tensor.to(device="cpu", dtype=self._momentum_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            debias_strength = group["debias_strength"]
            momentum_dtype = group.get("momentum_dtype", self._momentum_dtype)
            wd_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                buf_exp_avg, buf_exp_avg_sq, buf_p_fp32 = self._get_scratch_buffers(p)
                grad = p.grad.float()

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = self._new_cpu_state_tensor(p, momentum_dtype)
                    state["exp_avg_sq"] = self._new_cpu_state_tensor(p, momentum_dtype)

                state["step"] += 1
                step = state["step"]

                buf_exp_avg.copy_(state["exp_avg"], non_blocking=True)
                buf_exp_avg_sq.copy_(state["exp_avg_sq"], non_blocking=True)
                buf_p_fp32.copy_(p, non_blocking=True)

                buf_exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                buf_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                if debias_strength < 1.0:
                    bc1 = 1.0 - (1.0 - bc1) * debias_strength
                    bc2 = 1.0 - (1.0 - bc2) * debias_strength

                sqrt_bc2 = math.sqrt(bc2)
                step_size = lr / bc1

                if weight_decay != 0:
                    buf_p_fp32.mul_(wd_factor)

                denom = buf_exp_avg_sq.sqrt().div_(sqrt_bc2).add_(eps)
                buf_p_fp32.addcdiv_(buf_exp_avg, denom, value=-step_size)
                p.copy_(buf_p_fp32)

                state["exp_avg"].copy_(buf_exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(buf_exp_avg_sq, non_blocking=True)

        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["_momentum_dtype"] = self._momentum_dtype
        return state_dict

    def save_cpu_state(self):
        params_with_grad = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        cpu_state = {"_momentum_dtype": self._momentum_dtype}
        for i, p in enumerate(params_with_grad):
            if p in self.state:
                state = self.state[p]
                cpu_state[i] = {
                    "step": state.get("step", 0),
                    "exp_avg_cpu": state.get("exp_avg"),
                    "exp_avg_sq_cpu": state.get("exp_avg_sq"),
                }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        saved_dtype = cpu_state.get("_momentum_dtype", self._momentum_dtype)
        params_with_grad = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        for i, p in enumerate(params_with_grad):
            if i not in cpu_state:
                continue
            saved = cpu_state[i]
            exp_avg = saved.get("exp_avg", saved.get("exp_avg_cpu"))
            exp_avg_sq = saved.get("exp_avg_sq", saved.get("exp_avg_sq_cpu"))
            step = saved.get("step", 0)
            if torch.is_tensor(step):
                step = int(step.item())
            self.state[p] = {
                "step": step,
                "exp_avg": (
                    self._restore_cpu_state_tensor(exp_avg)
                    if exp_avg is not None else None
                ),
                "exp_avg_sq": (
                    self._restore_cpu_state_tensor(exp_avg_sq)
                    if exp_avg_sq is not None else None
                ),
            }

        if saved_dtype != self._momentum_dtype:
            print(
                f"[RavenAdamW] Loaded state saved in {saved_dtype}, "
                f"converted to {self._momentum_dtype}."
            )

    def load_state_dict(self, state_dict):
        saved_dtype = state_dict.pop("_momentum_dtype", torch.float32)

        if saved_dtype != self._momentum_dtype:
            print(
                f"[RavenAdamW] Loading state saved in {saved_dtype}, "
                f"but current optimizer uses {self._momentum_dtype}. "
                f"Converting..."
            )

        super().load_state_dict(state_dict)

        if saved_dtype != self._momentum_dtype:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad and p in self.state:
                        state = self.state[p]
                        if "exp_avg" in state:
                            state["exp_avg"] = state["exp_avg"].to(self._momentum_dtype)
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(self._momentum_dtype)

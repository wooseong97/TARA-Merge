import ipdb  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prob(nn.Module):
    def __init__(self, *, key_list, num, requires_grad, init_method="uniform"):
        super().__init__()
        self.num = num
        self.key_list = key_list
        self.key_map = {key: key.replace(".", "_") for key in key_list}
        self.reverse_key_map = {v: k for k, v in self.key_map.items()}

        # Make a dictionary of learnable logits per key
        self.logits = nn.ParameterDict(
            {
                self.key_map[key]: nn.Parameter(
                    self._init_coeff(key, num, init_method), requires_grad=requires_grad
                )
                for key in key_list
            }
        )

    def _forward_single_key(self, key, temperature=1.0, hard=False):
        if key not in self.logits:
            raise ValueError(f"Key '{key}' not found in logits.")

        logits = self.logits[key]
        # Softmax over logits gives a valid probability distribution
        probs = F.softmax(logits / temperature, dim=-1)

        if hard:
            # Return one-hot vector (e.g., for hard sampling)
            index = probs.argmax(dim=-1)
            one_hot = F.one_hot(index, num_classes=self.num).float()
            return one_hot
        return probs

    def forward(self, temperature=1.0, hard=False):
        # Forward pass for all keys
        return {
            key: self._forward_single_key(self.key_map[key], temperature, hard)
            for key in self.key_list
        }

    def _entropy_single_key(self, key):
        logits = self.logits[key]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs)

    def entropy(self):
        return {key: self._entropy_single_key(self.key_map[key]) for key in self.key_list}

    def sample(self, num_samples=1, temperature=1.0):
        # Gumbel-softmax sampling (differentiable)
        gumbel_noise = -torch.log(
            -torch.log(torch.rand((num_samples, self.num), device=self.logits.device) + 1e-9) + 1e-9
        )
        y = F.softmax((self.logits + gumbel_noise) / temperature, dim=-1)
        return y

    def _init_coeff(self, key, num, initialization):
        if initialization == "uniform":
            return torch.zeros(num)
        elif initialization == "stair":
            # Initialize logits in a stair-like manner
            # TODO: Fix later to support any number of adapters
            half_num = num // 2
            rank = half_num // 2
            stair = torch.cat(
                [
                    torch.ones(rank) * 5.0,
                    torch.ones(rank) * -5.0,
                    torch.ones(rank) * 5.0,
                    torch.ones(rank) * -5.0,
                ]
            )
            return stair
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")

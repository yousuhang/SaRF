import torch
import torch.nn as nn
import itertools

class ClassDistinctivenessLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)
        self.device = device

    def view_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    def forward(self, sal_tensor_list: [torch.Tensor]):
        loss_list = torch.Tensor([]).to(self.device)
        distance_list = torch.Tensor([]).to(self.device)
        for sal_comb in itertools.combinations(sal_tensor_list, 2):
            cos_sim = torch.abs(self.cossim(self.view_tensor(sal_comb[0]), self.view_tensor(sal_comb[1])))

            loss_list = torch.cat((loss_list, torch.unsqueeze(cos_sim.mean(), dim=0)))
            distance_list = torch.cat((distance_list, torch.unsqueeze(cos_sim, dim=1)), dim = 1)
        return torch.mean(loss_list), loss_list, distance_list

class PrelogitsSaliencySimLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)
        self.device = device

    def view_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    def forward(self, sal_tensor_list: [torch.Tensor], pre_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss_list = torch.Tensor([]).to(self.device)
        for i, sal in enumerate(sal_tensor_list):
            sign = (label != i)*2-1
            loss_list = torch.cat((loss_list, sign*torch.abs(self.cossim(self.view_tensor(sal), self.view_tensor(pre_logits)))))
        return torch.mean(loss_list)

class Norm2ZeroOne(nn.Module):
    def __init__(self, dim = 1):
        super().__init__()
        self.dim = dim
        self.relu_f = nn.ReLU()

    def forward(self, x):
        # x is output DTD
        x = self.relu_f(x)
        x = x / (torch.amax(x, dim=[1,2,3], keepdim = True) + 1e-12)
        return x


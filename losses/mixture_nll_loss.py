# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_csr

from losses.gaussian_nll_loss import GaussianNLLLoss
from losses.laplace_nll_loss import LaplaceNLLLoss
from losses.von_mises_nll_loss import VonMisesNLLLoss


class MixtureNLLLoss(nn.Module):

    def __init__(self,
                 component_distribution: Union[str, List[str]],
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(MixtureNLLLoss, self).__init__()
        self.reduction = reduction

        loss_dict = {
            'gaussian': GaussianNLLLoss,
            'laplace': LaplaceNLLLoss,
            'von_mises': VonMisesNLLLoss,
        }
        if isinstance(component_distribution, str):
            self.nll_loss = loss_dict[component_distribution](eps=eps, reduction='none')
        else:
            self.nll_loss = nn.ModuleList([loss_dict[dist](eps=eps, reduction='none')
                                           for dist in component_distribution])

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                prob: torch.Tensor,
                mask: torch.Tensor,
                ptr: Optional[torch.Tensor] = None,
                joint: bool = False) -> torch.Tensor:
        if isinstance(self.nll_loss, nn.ModuleList):
            nll = torch.cat(
                [self.nll_loss[i](pred=pred[..., [i, target.size(-1) + i]],
                                  target=target[..., [i]].unsqueeze(1))
                 for i in range(target.size(-1))],
                dim=-1)
        else:
            nll = self.nll_loss(pred=pred, target=target.unsqueeze(1))
        nll = (nll * mask.view(-1, 1, target.size(-2), 1)).sum(dim=(-2, -1))
        if joint:
            if ptr is None:
                nll = nll.sum(dim=0, keepdim=True)
            else:
                nll = segment_csr(src=nll, indptr=ptr, reduce='sum')
        else:
            pass
        log_pi = F.log_softmax(prob, dim=-1)
        loss = -torch.logsumexp(log_pi - nll, dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

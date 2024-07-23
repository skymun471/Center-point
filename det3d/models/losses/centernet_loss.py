import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')


    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos


class IDLoss(nn.Module):
  def __init__(self):
    super(IDLoss, self).__init__()

  def forward(self, matches, j_matches):
    # matches와 j_matches는 각각 [N, 2] 형태의 배열입니다.
    # 첫 번째 열은 detection ID, 두 번째 열은 track ID를 나타냅니다.

    # matches 및 j_matches를 딕셔너리로 변환하여 ID를 쉽게 찾을 수 있게 합니다.
    det_to_track = {det_id: track_id for det_id, track_id in matches}
    jpda_to_track = {det_id: track_id for det_id, track_id in j_matches}

    total_loss = 0.0
    count = 0

    # 각 detection ID가 대응하는 track ID와 일치하지 않는 경우에 대해 패널티를 부여합니다.
    for det_id, track_id in det_to_track.items():
      if det_id in jpda_to_track:
        jpda_track_id = jpda_to_track[det_id]
        if track_id != jpda_track_id:
          loss = F.mse_loss(torch.tensor(track_id).float(), torch.tensor(jpda_track_id).float(), reduction='mean')
          total_loss += loss
          count += 1

    if count == 0:
      return torch.tensor(0.0)

    return total_loss / count
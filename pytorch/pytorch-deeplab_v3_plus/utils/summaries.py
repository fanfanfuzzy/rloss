import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        
        pred_seg = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset)
        pred_seg_tensor = pred_seg.float() / 255.0
        grid_image = make_grid(pred_seg_tensor, 3, normalize=False)
        writer.add_image('Predicted label', grid_image, global_step)
        
        gt_seg = decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset)
        gt_seg_tensor = gt_seg.float() / 255.0
        grid_image = make_grid(gt_seg_tensor, 3, normalize=False)
        writer.add_image('Groundtruth label', grid_image, global_step)

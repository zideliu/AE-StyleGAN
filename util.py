import os
import sys
import ast
import math
import torch
import shutil
import random
import numpy as np
from torchvision.io import write_video
from torchvision import utils
from torch.nn import functional as F
from pathlib import Path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_log_dir(args):
    args.log_dir = os.path.join(args.log_root, args.name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'sample'), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'weight'), exist_ok=True)
    return args


def print_args(parser, args):
    message = f"Name: {getattr(args, 'name', 'NA')}\n"
    message += '--------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    exp_dir = args.log_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    file_name = os.path.join(exp_dir, 'args.txt')
    with open(file_name, 'wt') as f:
        f.write(message)
        f.write('\n')

    # save command to disk
    file_name = os.path.join(exp_dir, 'cmd.txt')
    with open(file_name, 'wt') as f:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('python ')
        f.write(' '.join(sys.argv))
        f.write('\n')

    # backup train code
    shutil.copyfile(sys.argv[0], os.path.join(args.log_dir, f'{os.path.basename(sys.argv[0])}.txt'))


def print_models(models, args):
    if not isinstance(models, (list, tuple)):
        models = [models]
    exp_dir = args.log_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    file_name = os.path.join(exp_dir, 'models.txt')
    with open(file_name, 'a+') as f:
        f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")
        for model in models:
            f.write(str(model))
            f.write("\n\n")


def str2list(attr_bins):
    assert (isinstance(attr_bins, str))
    attr_bins = attr_bins.strip()
    if attr_bins.endswith(('.npy', '.npz')):
        attr_bins = np.load(attr_bins)
    else:
        assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
        # attr_bins = np.array(ast.literal_eval(attr_bins))
        attr_bins = ast.literal_eval(attr_bins)
    return attr_bins


def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    :param v:
    :return: bool(v)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def linspace(idx_range, val_range, idx_max, val_default=None):
    if len(idx_range) >= 2 and len(val_range) >= 2:
        dtype = np.array(val_range).dtype
        val_list = (
            [val_range[0]] * max(0, int(idx_range[0])) + 
            list(np.linspace(val_range[0], val_range[1],
                idx_range[1] - idx_range[0] + 1, dtype=dtype)) + 
            [val_range[1]] * max(0, int(idx_max - idx_range[1] + 2))
        )
    else:
        val_list = [val_default] * (idx_max + 1)
    return val_list


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg, path, nrow=int(n_sample ** 0.5), normalize=True, value_range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video+1.)/2.*255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


def estimate_optical_flow(netNetwork, tenFirst, tenSecond):
    # Copied from https://github.com/sniklaus/pytorch-pwc/blob/master/run.py
    # Assume tensors are normalized to [-1, 1]
    tenFirst = (tenFirst + 1.) / 2
    tenSecond = (tenSecond + 1.) / 2
    assert(tenFirst.shape[1] == tenSecond.shape[1])
    assert(tenFirst.shape[2] == tenSecond.shape[2])

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = min(int(math.floor(math.ceil(intWidth / 64.0) * 64.0)), 128)
    intPreprocessedHeight = min(int(math.floor(math.ceil(intHeight / 64.0) * 64.0)), 128)

    tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    # tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    tenFlow = 20.0 * netNetwork(tenPreprocessedFirst, tenPreprocessedSecond)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :]


def randperm(n, ordered=False):
    # ordered: include ordered permutation?
    if ordered:
        return torch.randperm(n)
    else:
        perm_ord = torch.tensor(range(n))
        while True:
            perm = torch.randperm(n)
            if (perm != perm_ord).any():
                return perm        


def permute_dim(tensor, i=0, j=1, ordered=False):
    # Permute along dim i for each j.
    # e.g.: Factor-VAE, i = 0, j = 1; Jigsaw, i = 2, j = 0
    device = tensor.device
    n = tensor.shape[i]
    return torch.cat([torch.index_select(t, i, randperm(n, ordered).to(device)) for t in tensor.split(1, j)], j)


"""
Negative Data Augmentations
"""
def negative_augment(img, nda_type='jigsaw_4'):
    img_aug = None
    if nda_type.startswith('jigsaw'):
        n, c, h, w = img.shape
        n_patch = int(nda_type.split('_')[1])  # number of patches
        n_patch_sqrt = int(n_patch ** 0.5)
        h_patch, w_patch = h//n_patch_sqrt, w//n_patch_sqrt
        patches = F.unfold(img, kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch))
        patches_perm = permute_dim(patches, 2, 0)
        img_aug = F.fold(patches_perm, (h, w), kernel_size=(h_patch, w_patch), stride=(h_patch, w_patch))
    return img_aug, None

try:
    import wandb
except ImportError:
    wandb = None
import logging


def log(output, flush=True):
    logging.info(output)
    if flush:
        print(output)

def set_up_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id

def set_up_wandb(args):
    if wandb is not None:
        # name = Path(args.log_dir).name
        name = args.name
        resume = getattr(args, 'resume', False)
        run_id = set_up_wandb_run_id(args.log_dir, resume)
        args.wandb_run_id = run_id
        run = wandb.init(
            project=getattr(args, 'wandb_project', 'stylegan2'),
            name=name,
            id=run_id,
            config=args,
            resume=True if resume else "allow",
        )
        return run
    else:
        log_str = "Failed to set up wandb - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)

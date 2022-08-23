import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import util
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from idinvert_pytorch.models.perceptual_model import VGG16
from dataset import MultiResolutionDataset, VideoFolderDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    # Endless iterator
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def accumulate_batches(data_iter, num):
    samples = []
    while num > 0:
        data, _ = next(data_iter)
        samples.append(data)
        num -= data.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def train(args, loader, encoder, generator, discriminator, vggnet, pwcnet, e_optim, d_optim, e_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    e_loss_val = 0
    rec_loss_val = 0
    vgg_loss_val = 0
    adv_loss_val = 0
    loss_dict = {"d": torch.tensor(0., device=device),
                 "real_score": torch.tensor(0., device=device),
                 "fake_score": torch.tensor(0., device=device),
                 "r1_d": torch.tensor(0., device=device),
                 "r1_e": torch.tensor(0., device=device),
                 "rec": torch.tensor(0., device=device),}
    avg_pix_loss = util.AverageMeter()
    avg_vgg_loss = util.AverageMeter()

    if args.distributed:
        e_module = encoder.module
        d_module = discriminator.module
        g_module = generator.module
    else:
        e_module = encoder
        d_module = discriminator
        g_module = generator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_x = accumulate_batches(loader, args.n_sample).to(device)

    requires_grad(generator, False)  # always False
    generator.eval()  # Generator should be ema and in eval mode

    # if args.no_ema or e_ema is None:
    #     e_ema = encoder
    
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img1, real_img2 = next(loader)
        real_img1 = real_img1.to(device)
        real_img2 = real_img2.to(device)

        # Train Encoder
        if args.toggle_grads:
            requires_grad(encoder, True)
            requires_grad(discriminator, False)
        pix_loss = vgg_loss = adv_loss = rec_loss = torch.tensor(0., device=device)
        latent_real1 = encoder(real_img1)
        fake_img1, _ = generator([latent_real1], input_is_latent=True, return_latents=False)
        latent_real2 = encoder(real_img2)
        fake_img2, _ = generator([latent_real2], input_is_latent=True, return_latents=False)

        if args.lambda_adv > 0:
            # if args.augment:
            #     fake_img_aug1, _ = augment(fake_img1, ada_aug_p)
            # else:
            #     fake_img_aug1 = fake_img1
            fake_img_pair = torch.cat((fake_img1, fake_img2), 1)
            fake_pred = discriminator(fake_img_pair)
            adv_loss = g_nonsaturating_loss(fake_pred)

        if args.lambda_pix > 0:
            pix_loss = torch.mean((real_img1 - fake_img1) ** 2)
            if args.reconstruct_pair:
                pix_loss += torch.mean((real_img2 - fake_img2) ** 2)

        if args.lambda_vgg > 0:
            real_feat = vggnet(real_img1)
            fake_feat = vggnet(fake_img1)
            vgg_loss = torch.mean((real_feat - fake_feat) ** 2)
            if args.reconstruct_pair:
                real_feat = vggnet(real_img2)
                fake_feat = vggnet(fake_img2)
                vgg_loss += torch.mean((real_feat - fake_feat) ** 2)

        e_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg + adv_loss * args.lambda_adv

        loss_dict["e"] = e_loss
        loss_dict["pix"] = pix_loss
        loss_dict["vgg"] = vgg_loss
        loss_dict["adv"] = adv_loss

        encoder.zero_grad()
        e_loss.backward()
        e_optim.step()

        # if args.train_on_fake:
        #     e_regularize = args.e_rec_every > 0 and i % args.e_rec_every == 0
        #     if e_regularize and args.lambda_rec > 0:
        #         noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        #         fake_img, latent_fake = generator(noise, input_is_latent=False, return_latents=True)
        #         latent_pred = encoder(fake_img)
        #         if latent_pred.ndim < 3:
        #             latent_pred = latent_pred.unsqueeze(1).repeat(1, latent_fake.size(1), 1)
        #         rec_loss = torch.mean((latent_fake - latent_pred) ** 2)
        #         encoder.zero_grad()
        #         (rec_loss * args.lambda_rec).backward()
        #         e_optim.step()
        #         loss_dict["rec"] = rec_loss

        e_regularize = args.e_reg_every > 0 and i % args.e_reg_every == 0
        if e_regularize:
            # why not regularize on augmented real?
            real_img_pair = torch.cat((real_img1, real_img2), 1)
            real_img_pair.requires_grad = True
            real_pred = encoder(real_img_pair)
            r1_loss_e = d_r1_loss(real_pred, real_img_pair)

            encoder.zero_grad()
            (args.r1 / 2 * r1_loss_e * args.e_reg_every + 0 * real_pred.view(-1)[0]).backward()
            e_optim.step()

            loss_dict["r1_e"] = r1_loss_e

        if not args.no_ema and e_ema is not None:
            accumulate(e_ema, e_module, accum)
        
        # Train Discriminator
        if args.toggle_grads:
            requires_grad(encoder, False)
            requires_grad(discriminator, True)
        if not args.no_update_discriminator and args.lambda_adv > 0:
            latent_real1 = encoder(real_img1)
            fake_img1, _ = generator([latent_real1], input_is_latent=True, return_latents=False)
            latent_real2 = encoder(real_img2)
            fake_img2, _ = generator([latent_real2], input_is_latent=True, return_latents=False)

            # if args.augment:
            #     real_img_aug, _ = augment(real_img, ada_aug_p)
            #     fake_img_aug, _ = augment(fake_img, ada_aug_p)
            # else:
            #     real_img_aug = real_img
            #     fake_img_aug = fake_img
            
            fake_img_pair = torch.cat((fake_img1, fake_img2), 1)
            real_img_pair = torch.cat((real_img1, real_img2), 1)

            fake_pred = discriminator(fake_img_pair)
            real_pred = discriminator(real_img_pair)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            # if args.augment and args.augment_p == 0:
            #     ada_aug_p = ada_augment.tune(real_pred)
            #     r_t_stat = ada_augment.r_t_stat
            
            d_regularize = args.d_reg_every > 0 and i % args.d_reg_every == 0
            if d_regularize:
                # why not regularize on augmented real?
                real_img_pair = torch.cat((real_img1, real_img2), 1)
                real_img_pair.requires_grad = True
                real_pred = discriminator(real_img_pair)
                r1_loss_d = d_r1_loss(real_pred, real_img_pair)

                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss_d * args.d_reg_every + 0 * real_pred.view(-1)[0]).backward()
                # Why 0* ? Answer is here https://github.com/rosinality/stylegan2-pytorch/issues/76
                d_optim.step()

                loss_dict["r1_d"] = r1_loss_d

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        e_loss_val = loss_reduced["e"].mean().item()
        r1_d_val = loss_reduced["r1_d"].mean().item()
        r1_e_val = loss_reduced["r1_e"].mean().item()
        pix_loss_val = loss_reduced["pix"].mean().item()
        vgg_loss_val = loss_reduced["vgg"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        rec_loss_val = loss_reduced["rec"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img1.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img1.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; e: {e_loss_val:.4f}; r1_d: {r1_d_val:.4f}; r1_e: {r1_e_val:.4f}; "
                    f"pix: {pix_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; adv: {adv_loss_val:.4f}; "
                    f"rec: {rec_loss_val:.4f}; augment: {ada_aug_p:.4f}"
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    latent_x = e_ema(sample_x)
                    fake_x, _ = generator([latent_x], input_is_latent=True, return_latents=False)
                    sample_pix_loss = torch.sum((sample_x - fake_x) ** 2)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(f"{i:07d}; pix: {avg_pix_loss.avg}; vgg: {avg_vgg_loss.avg}; "
                            f"ref: {sample_pix_loss.item()};\n")

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Encoder": e_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1 D": r1_d_val,
                        "R1 E": r1_e_val,
                        "Pix Loss": pix_loss_val,
                        "VGG Loss": vgg_loss_val,
                        "Adv Loss": adv_loss_val,
                        "Rec Loss": rec_loss_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                    }
                )

            if i % args.log_every == 0:
                with torch.no_grad():
                    e_eval = encoder if args.no_ema else e_ema
                    e_eval.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x.shape)[1:]
                    latent_real = e_eval(sample_x)
                    fake_img, _ = generator([latent_real], input_is_latent=True, return_latents=False)
                    sample = torch.cat((sample_x.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}.png"),
                        nrow=nrow,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    e_eval.train()

            if i % args.save_every == 0:
                e_eval = encoder if args.no_ema else e_ema
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"{str(i).zfill(6)}.pt"),
                )
            
            if i % args.save_latest_every == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_module.state_dict(),
                        "e_ema": e_eval.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"latest.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 encoder trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default='local.db')
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=100, help="save latest checkpoints every # iters")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--no_update_discriminator", action='store_true')
    parser.add_argument("--no_load_discriminator", action='store_true')
    parser.add_argument("--toggle_grads", action='store_true')
    parser.add_argument("--use_optical_flow", action='store_true')
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--no_ema", action='store_true', help="do not use ema if enabled")
    parser.add_argument("--train_on_fake", action='store_true', help="train encoder on fake?")
    parser.add_argument("--e_rec_every", type=int, default=1, help="interval of minimizing recon loss on w")
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="recon loss on style (w)")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--vgg_ckpt", type=str, default="pretrained/vgg16.pth")
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_tied')
    parser.add_argument("--stddev_group", type=int, default=4)
    parser.add_argument("--reconstruct_pair", action='store_true')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--e_reg_every",
        type=int,
        default=0,
        help="interval of the applying r1 regularization, no if 0",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization, no if 0",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()
    util.seed_everything()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_latent = int(np.log2(args.size)) * 2 - 2  # used in Generator
    args.latent = 512  # fixed, dim of w or z (same size)
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError
    args.n_mlp = 8

    args.start_iter = 0
    # args.mixing = 0  # no mixing
    util.set_log_dir(args)
    util.print_args(parser, args)
    
    # Auxiliary models (VGG and PWC)
    vggnet = VGG16(output_layer_idx=args.output_layer_idx).to(device)
    vgg_ckpt = torch.load(args.vgg_ckpt, map_location=lambda storage, loc: storage)
    vggnet.load_state_dict(vgg_ckpt)

    pwcnet = None
    if args.use_optical_flow:
        pwc = __import__('pytorch-pwc.run', globals(), locals(), ['Network'], 0)
        pwcnet = pwc.Network().to(device)  # state_dict loaded in init
        pwcnet.eval()

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, in_channel=6,
    ).to(device)
    # generator = Generator(
    #     args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    # ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    # accumulate(g_ema, generator, 0)

    e_ema = None
    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        encoder = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=True,
            use_wscale=args.use_wscale).to(device)
        if not args.no_ema:
            e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
                which_latent=args.which_latent, reshape_latent=True,
                use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=True, stddev_group=args.stddev_group).to(device)
        if not args.no_ema:
            e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
                which_latent=args.which_latent, reshape_latent=True, stddev_group=args.stddev_group).to(device)
    if not args.no_ema:
        e_ema.eval()
        accumulate(e_ema, encoder, 0)

    # For lazy regularization (see paper appendix page 11)
    e_reg_ratio = args.e_reg_every / (args.e_reg_every + 1) if args.e_reg_every > 0 else 1.
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1.

    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        # generator.load_state_dict(ckpt["g"])
        if 'g_ema' in ckpt:
            g_ema.load_state_dict(ckpt["g_ema"])
        else:
            g_ema.load_state_dict(ckpt["g"])
        
        # if not args.no_load_discriminator:
        #     discriminator.load_state_dict(ckpt["d"])
        #     d_optim.load_state_dict(ckpt["d_optim"])

        if args.resume:
            try:
                ckpt_name = os.path.basename(args.ckpt)
                if 'latest' in ckpt_name and 'iter' in ckpt:
                    args.start_iter = ckpt["iter"]
                else:
                    args.start_iter = int(os.path.splitext(ckpt_name)[0])
            except ValueError:
                pass
            encoder.load_state_dict(ckpt["e"])
            e_optim.load_state_dict(ckpt["e_optim"])

    if args.distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    assert(args.dataset == 'videofolder')
    # [Note] Potentially, same transforms will be applied to a batch of images,
    # either a sequence or a pair (optical flow), so we should apply ToTensor first.
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),  # this should be done in loader
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.size),  # Image.LANCZOS
            transforms.CenterCrop(args.size),
            # transforms.ToTensor(),  # normally placed here
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = VideoFolderDataset(args.path, transform, mode='nframe', nframe_num=2, cache=args.cache)
    if len(dataset) == 0:
        raise ValueError
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)
    assert(not args.augment)  # currently only supports no augment
    train(args, loader, encoder, g_ema, discriminator, vggnet, pwcnet, e_optim, d_optim, e_ema, device)

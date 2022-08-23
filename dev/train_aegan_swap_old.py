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
from torchvision import datasets, transforms, utils
from PIL import Image
from tqdm import tqdm
import util
from calc_inception import load_patched_inception_v3
from fid import extract_feature_from_samples, calc_fid
import pickle
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
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


def sample_data2(loader):
    # image and label pair
    while True:
        for batch, _ in loader:
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
        imgs = next(data_iter)
        samples.append(imgs)
        num -= imgs.size(0)
    samples = torch.cat(samples, dim=0)
    if num < 0:
        samples = samples[:num, ...]
    return samples


def load_real_samples(args, data_iter):
    npy_path = args.sample_cache
    if npy_path is not None and os.path.exists(npy_path):
        sample_x = torch.from_numpy(np.load(npy_path)).to(args.device)
    else:
        sample_x = accumulate_batches(data_iter, args.n_sample).to(args.device)
        if npy_path is not None:
            np.save(npy_path, sample_x.cpu().numpy())
    return sample_x


def get_batch(loader, device):
    frames = next(loader)  # [N, T, C, H, W]
    batch = frames.shape[0]
    nframe_num = frames.shape[1]
    frames1 = frames[:,0,...]
    frames2 = frames[range(batch),torch.randint(1,nframe_num,(batch,)),...]
    frames1 = frames1.to(device)
    frames2 = frames2.to(device)
    return frames1, frames2


def train(args, loader, generator, encoder, discriminator, discriminator2,
          vggnet, g_optim, e_optim, d_optim, d2_optim, g_ema, e_ema, device):
    # kwargs_d = {'detach_aux': args.detach_d_aux_head}
    if args.dataset == 'imagefolder':
        loader = sample_data2(loader)
    else:
        loader = sample_data(loader)
    
    if args.eval_every > 0:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()
        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
    else:
        inception = real_mean = real_cov = None
    mean_latent = None

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}
    avg_pix_loss = util.AverageMeter()
    avg_vgg_loss = util.AverageMeter()

    if args.distributed:
        g_module = generator.module
        e_module = encoder.module
        d_module = discriminator.module
    else:
        g_module = generator
        e_module = encoder
        d_module = discriminator
    
    d2_module = None
    if discriminator2 is not None:
        if args.distributed:
            d2_module = discriminator2.module
        else:
            d2_module = discriminator2

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_x = load_real_samples(args, loader)
    sample_x1 = sample_x[:,0,...]
    sample_x2 = sample_x[:,-1,...]
    sample_idx = torch.randperm(args.n_sample)

    n_step_max = max(args.n_step_d, args.n_step_e)

    requires_grad(g_ema, False)
    requires_grad(e_ema, False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        frames = [get_batch(loader, device) for _ in range(n_step_max)]

        # Train Discriminator
        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(discriminator, True)
        for step_index in range(args.n_step_d):
            frames1, frames2 = frames[step_index]
            real_img = frames1
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            if args.use_ema:
                g_ema.eval()
                fake_img, _ = g_ema(noise)
            else:
                fake_img, _ = generator(noise)
            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)
                fake_img, _ = augment(fake_img, ada_aug_p)
            else:
                real_img_aug = real_img
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img_aug)
            d_loss_fake = F.softplus(fake_pred).mean()
            d_loss_real = F.softplus(-real_pred).mean()
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            d_loss_rec = 0.
            if args.lambda_rec_d > 0 and not args.decouple_d:  # Do not train D on x_rec if decouple_d
                if args.use_ema:
                    e_ema.eval()
                    g_ema.eval()
                    latent_real, _ = e_ema(real_img)
                    rec_img, _ = g_ema([latent_real], input_is_latent=True)
                else:
                    latent_real, _ = encoder(real_img)
                    rec_img, _ = generator([latent_real], input_is_latent=True)
                rec_pred = discriminator(rec_img)
                d_loss_rec = F.softplus(rec_pred).mean()
                loss_dict["rec_score"] = rec_pred.mean()

            d_loss_cross = 0.
            if args.lambda_cross_d > 0 and not args.decouple_d:
                if args.use_ema:
                    e_ema.eval()
                    w1, _ = e_ema(frames1)
                    w2, _ = e_ema(frames2)
                else:
                    w1, _ = encoder(frames1)
                    w2, _ = encoder(frames2)
                dw = w2 - w1
                dw_shuffle = dw[torch.randperm(args.batch),...]
                if args.use_ema:
                    g_ema.eval()
                    cross_img, _ = g_ema([w1 + dw_shuffle], input_is_latent=True)
                else:
                    cross_img, _ = generator([w1 + dw_shuffle], input_is_latent=True)
                cross_pred = discriminator(cross_img)
                d_loss_cross = F.softplus(cross_pred).mean()
            
            d_loss_fake_cross = 0.
            if args.lambda_fake_cross_d > 0:
                if args.use_ema:
                    e_ema.eval()
                    w1, _ = e_ema(frames1)
                    w2, _ = e_ema(frames2)
                else:
                    w1, _ = encoder(frames1)
                    w2, _ = encoder(frames2)
                dw = w2 - w1
                noise = mixing_noise(args.batch, args.latent, args.mixing, device)
                if args.use_ema:
                    g_ema.eval()
                    style = g_ema.get_styles(noise).view(args.batch, -1)
                else:
                    style = generator.get_styles(noise).view(args.batch, -1)
                if dw.shape[1] < style.shape[1]:  # W space
                    dw = dw.repeat(1, args.n_latent)
                if args.use_ema:
                    cross_img, _ = g_ema([style + dw], input_is_latent=True)
                else:
                    cross_img, _ = generator([style + dw], input_is_latent=True)
                fake_cross_pred = discriminator(cross_img)
                d_loss_fake_cross = F.softplus(fake_cross_pred).mean()

            d_loss = (d_loss_real + d_loss_fake + d_loss_fake_cross * args.lambda_fake_cross_d + 
                d_loss_rec * args.lambda_rec_d + d_loss_cross * args.lambda_cross_d)
            loss_dict["d"] = d_loss

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()
        loss_dict["r1"] = r1_loss
        
        # Train Discriminator2
        if args.decouple_d and discriminator2 is not None:
            requires_grad(generator, False)
            requires_grad(encoder, False)
            requires_grad(discriminator2, True)
            for step_index in range(args.n_step_e):  # n_step_d2 is same as n_step_e
                frames1, frames2 = frames[step_index]
                real_img = frames1
                if args.use_ema:
                    e_ema.eval()
                    g_ema.eval()
                    latent_real, _ = e_ema(real_img)
                    rec_img, _ = g_ema([latent_real], input_is_latent=True)
                else:
                    latent_real, _ = encoder(real_img)
                    rec_img, _ = generator([latent_real], input_is_latent=True)
                rec_pred = discriminator2(rec_img)
                d2_loss_rec = F.softplus(rec_pred).mean()
                real_pred1 = discriminator2(frames1)
                d2_loss_real = F.softplus(-real_pred1).mean()
                if args.use_frames2_d:
                    real_pred2 = discriminator2(frames2)
                    d2_loss_real += F.softplus(-real_pred2).mean()

                if args.use_ema:
                    e_ema.eval()
                    w1, _ = e_ema(frames1)
                    w2, _ = e_ema(frames2)
                else:
                    w1, _ = encoder(frames1)
                    w2, _ = encoder(frames2)
                dw = w2 - w1
                dw_shuffle = dw[torch.randperm(args.batch),...]
                cross_img, _ = generator([w1 + dw_shuffle], input_is_latent=True)
                cross_pred = discriminator2(cross_img)
                d2_loss_cross = F.softplus(cross_pred).mean()

                d2_loss = d2_loss_real + d2_loss_rec + d2_loss_cross
                loss_dict["d2"] = d2_loss
                loss_dict["rec_score"] = rec_pred.mean()
                loss_dict["cross_score"] = cross_pred.mean()

                discriminator2.zero_grad()
                d2_loss.backward()
                d2_optim.step()

            d_regularize = i % args.d_reg_every == 0
            if d_regularize:
                real_img.requires_grad = True
                real_pred = discriminator2(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)
                discriminator2.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
                d2_optim.step()

        # Train Encoder
        requires_grad(encoder, True)
        requires_grad(generator, args.train_ge)
        requires_grad(discriminator, False)
        if discriminator2 is not None:
            requires_grad(discriminator2, False)
        pix_loss = vgg_loss = adv_loss = torch.tensor(0., device=device)
        for step_index in range(args.n_step_e):
            frames1, frames2 = frames[step_index]
            real_img = frames1
            latent_real, _ = encoder(real_img)
            if args.use_ema:
                g_ema.eval()
                rec_img, _ = g_ema([latent_real], input_is_latent=True)
            else:
                rec_img, _ = generator([latent_real], input_is_latent=True)
            if args.lambda_adv > 0:
                if not args.decouple_d:
                    rec_pred = discriminator(rec_img)
                else:
                    rec_pred = discriminator2(rec_img)
                adv_loss = g_nonsaturating_loss(rec_pred)
            if args.lambda_pix > 0:
                if args.pix_loss == 'l2':
                    pix_loss = torch.mean((rec_img - real_img) ** 2)
                else:
                    pix_loss = F.l1_loss(rec_img, real_img)
            if args.lambda_vgg > 0:
                vgg_loss = torch.mean((vggnet(real_img) - vggnet(rec_img)) ** 2)
            
            e_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg + adv_loss * args.lambda_adv
            loss_dict["e"] = e_loss
            loss_dict["pix"] = pix_loss
            loss_dict["vgg"] = vgg_loss
            loss_dict["adv"] = adv_loss

            if args.train_ge:
                encoder.zero_grad()
                generator.zero_grad()
                e_loss.backward()
                e_optim.step()
                g_optim.step()
            else:
                encoder.zero_grad()
                e_loss.backward()
                e_optim.step()

        # Train Generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        if discriminator2 is not None:
            requires_grad(discriminator2, False)
        frames1, frames2 = frames[0]
        real_img = frames1
        g_loss_fake = 0.
        if args.lambda_fake_g > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            g_loss_fake = g_nonsaturating_loss(fake_pred)

        g_loss_rec = 0.
        if args.lambda_rec_g > 0:
            if args.use_ema:
                e_ema.eval()
                latent_real, _ = e_ema(real_img)
            else:
                latent_real, _ = encoder(real_img)
            rec_img, _ = generator([latent_real], input_is_latent=True)
            if not args.decouple_d:
                rec_pred = discriminator(rec_img)
            else:
                rec_pred = discriminator2(rec_img)
            g_loss_rec = g_nonsaturating_loss(rec_pred)

        g_loss_cross = 0.
        if args.lambda_cross_g > 0:
            if args.use_ema:
                e_ema.eval()
                w1, _ = e_ema(frames1)
                w2, _ = e_ema(frames2)
            else:
                w1, _ = encoder(frames1)
                w2, _ = encoder(frames2)
            dw = w2 - w1
            dw_shuffle = dw[torch.randperm(args.batch),...]
            cross_img, _ = generator([w1 + dw_shuffle], input_is_latent=True)
            if not args.decouple_d:
                cross_pred = discriminator(cross_img)
            else:
                cross_pred = discriminator2(cross_img)
            g_loss_cross = g_nonsaturating_loss(cross_pred)

        g_loss_fake_cross = 0.
        if args.lambda_fake_cross_g > 0:
            if args.use_ema:
                e_ema.eval()
                w1, _ = e_ema(frames1)
                w2, _ = e_ema(frames2)
            else:
                w1, _ = encoder(frames1)
                w2, _ = encoder(frames2)
            dw = w2 - w1
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            style = generator.get_styles(noise).view(args.batch, -1)
            if dw.shape[1] < style.shape[1]:  # W space
                dw = dw.repeat(1, args.n_latent)
            cross_img, _ = generator([style + dw], input_is_latent=True)
            fake_cross_pred = discriminator(cross_img)
            g_loss_fake_cross = g_nonsaturating_loss(fake_cross_pred)

        g_loss = (g_loss_fake * args.lambda_fake_g + g_loss_rec * args.lambda_rec_g + 
            g_loss_cross * args.lambda_cross_g + g_loss_fake_cross * args.lambda_fake_cross_g)
        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = args.g_reg_every > 0 and i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )
            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
            weighted_path_loss.backward()
            g_optim.step()
            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        pix_loss_val = loss_reduced["pix"].mean().item()
        vgg_loss_val = loss_reduced["vgg"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; "
                    f"pix: {pix_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; adv: {adv_loss_val:.4f}"
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    latent_x, _ = e_ema(sample_x)
                    fake_x, _ = generator([latent_x], input_is_latent=True, return_latents=False)
                    sample_pix_loss = torch.sum((sample_x - fake_x) ** 2)
                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(f"{i:07d}; pix: {avg_pix_loss.avg}; vgg: {avg_vgg_loss.avg}; "
                            f"ref: {sample_pix_loss.item()};\n")

            if args.eval_every > 0 and i % args.eval_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    if args.truncation < 1:
                        mean_latent = g_ema.mean_latent(4096)
                    features = extract_feature_from_samples(
                        g_ema, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device
                    ).numpy()
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)
                    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                print("fid:", fid)
                with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                    f.write(f"{i:07d}; fid: {float(fid):.4f};\n")

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.log_every == 0:
                with torch.no_grad():
                    # Fixed fake samples
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    # Reconstruction samples
                    e_ema.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x.shape)[1:]
                    latent_real, _ = e_ema(sample_x)
                    fake_img, _ = g_ema([latent_real], input_is_latent=True, return_latents=False)
                    sample = torch.cat((sample_x.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        fake_img.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    utils.save_image(
                        sample.reshape(2*args.n_sample, *nchw),
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-recon.png"),
                        nrow=nrow,
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % args.save_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"{str(i).zfill(6)}.pt"),
                )
            
            if i % args.save_latest_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "d2": d2_module.state_dict() if args.decouple_d else None,
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "d2_optim": d2_optim.state_dict() if args.decouple_d else None,
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,
                    },
                    os.path.join(args.log_dir, 'weight', f"latest.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=100, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=100, help="save latest checkpoints every # iters")
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
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_plus')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_rec_d", type=float, default=1.0)
    parser.add_argument("--lambda_fake_d", type=float, default=1.0)
    parser.add_argument("--lambda_cross_d", type=float, default=1.0)
    parser.add_argument("--lambda_fake_cross_d", type=float, default=0)
    parser.add_argument("--lambda_rec_g", type=float, default=0)
    parser.add_argument("--lambda_cross_g", type=float, default=0)
    parser.add_argument("--lambda_fake_cross_g", type=float, default=0)
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--train_ge", action='store_true', help="update generator with encoder")
    parser.add_argument("--inception", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=50000, help="number of the samples for calculating FID")
    parser.add_argument("--use_balanced_d_loss", action='store_true', help="adjust discriminator loss weights?")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--debug", type=str, default='none')
    parser.add_argument("--decouple_d", action='store_true')
    parser.add_argument("--use_ema", action='store_true')
    parser.add_argument("--use_frames2_d", action='store_true')
    parser.add_argument("--n_step_d", type=int, default=1)
    parser.add_argument("--n_step_e", type=int, default=1)
    parser.add_argument("--nframe_num", type=int, default=5)

    args = parser.parse_args()
    util.seed_everything()
    args.device = device

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.n_mlp = 8
    args.n_latent = int(np.log2(args.size)) * 2 - 2
    args.latent = 512
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError

    args.start_iter = 0
    util.set_log_dir(args)
    util.print_args(parser, args)

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) if args.g_reg_every > 0 else 1.
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1) if args.d_reg_every > 0 else 1.

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    discriminator2 = d2_optim = None
    if args.decouple_d:
        discriminator2 = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,
        ).to(device)
        d2_optim = optim.Adam(
            discriminator2.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    # Define Encoder
    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        encoder = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=True,
            use_wscale=args.use_wscale).to(device)
        e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, reshape_latent=True,
            use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=True, stddev_group=args.stddev_group).to(device)
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, reshape_latent=True, stddev_group=args.stddev_group).to(device)
    e_ema.eval()
    accumulate(e_ema, encoder, 0)
    
    e_reg_ratio = 1.
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr * e_reg_ratio,
        betas=(0 ** e_reg_ratio, 0.99 ** e_reg_ratio),
    )

    from idinvert_pytorch.models.perceptual_model import VGG16
    vggnet = VGG16(output_layer_idx=args.output_layer_idx).to(device)
    vgg_ckpt = torch.load(args.vgg_ckpt, map_location=lambda storage, loc: storage)
    vggnet.load_state_dict(vgg_ckpt)

    if args.resume and args.ckpt is None:
        args.ckpt = os.path.join(args.log_dir, 'weight', f"latest.pt")
    if args.ckpt is not None:  # resume
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if 'iter' in ckpt:
                args.start_iter = ckpt["iter"]
            else:
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        encoder.load_state_dict(ckpt["e"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        e_ema.load_state_dict(ckpt["e_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        e_optim.load_state_dict(ckpt["e_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        if discriminator2 is not None:
            discriminator2.load_state_dict(ckpt["d2"])
            d2_optim.load_state_dict(ckpt["d2_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
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
        if discriminator2 is not None:
            discriminator2 = nn.parallel.DistributedDataParallel(
                discriminator2,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )
    dataset = None
    if args.dataset == 'videofolder':
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
        dataset = VideoFolderDataset(args.path, transform, mode='nframe',
            nframe_num=args.nframe_num, cache=args.cache, unbind=False,
        )
        if len(dataset) == 0:
            raise ValueError
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.name)

    train(args, loader, generator, encoder, discriminator, discriminator2,
          vggnet, g_optim, e_optim, d_optim, d2_optim, g_ema, e_ema, device)

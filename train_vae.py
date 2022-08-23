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
from fid import extract_feature_from_samples, calc_fid, extract_feature_from_reconstruction
import pickle
import pdb
st = pdb.set_trace

try:
    import wandb

except ImportError:
    wandb = None

from dataset import get_image_dataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

from torch.distributions import Normal, kl_divergence, Bernoulli
LIKELIHOOD_SIGMA = np.sqrt(0.5)

def scale_grad(model, scale):
    if model is not None:
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    if model is not None:
        for p in model.parameters():
            p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    # Endless image iterator
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield batch[0]
            else:
                yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
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


def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
    if last_layer is None:
        return 1.0
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)
    if isinstance(last_layer, (list, tuple)):
        nll_grads = torch.cat([v.view(-1) for v in nll_grads])
        g_grads = torch.cat([v.view(-1) for v in g_grads])
    else:
        nll_grads = nll_grads[0]
        g_grads = g_grads[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


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
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
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

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def log_mean_exp(x, dim):
    m = x.max(dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))

def log_likelihood(encoder, decoder, imgs):
    w_mean, w_logvar = encoder(imgs)
    w_std = torch.exp(0.5 * w_logvar)
    w_sample = reparameterize(w_mean, w_logvar)
    x_recon, _ = decoder([w_sample], input_is_latent=True)
    x_recon = torch.clamp(x_recon, -1.0, 1.0)
    log_q = Normal(w_mean, w_std).log_prob(w_sample)
    log_p = Normal(0, 1).log_prob(w_sample)
    p_x_given_z = Normal(x_recon, LIKELIHOOD_SIGMA)
    log_like = p_x_given_z.log_prob(imgs)
    log_iw = log_p.flatten(start_dim=1).sum(-1) - log_q.flatten(start_dim=1).sum(-1)
    return log_like.flatten(start_dim=1).sum(-1), log_iw


@torch.no_grad()
def evaluate_nll(encoder, decoder, data_loader, opt=None):
    ll_list = list()
    n_batch = 0
    for imgs in data_loader:
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        imgs = imgs.cuda()
        ll_sample_list = list()
        for _ in range(opt.nll_num_samples):
            log_like, log_iw = log_likelihood(encoder, decoder, imgs)
            ll_sample_list.append((log_like + log_iw).detach().cpu())
        ll_sample = log_mean_exp(torch.stack(ll_sample_list, dim=1), dim=1)
        ll_list.append(ll_sample)
        n_batch += 1
        if opt.nll_num_batches > 0 and n_batch >= opt.nll_num_batches:
            break
    ll_all = torch.cat(ll_list, dim=0)
    # metrics = {'nll': -ll_all.mean(0).item()}
    # return metrics
    return -ll_all.mean(0).item()

def train(args, loader, loader2, generator, encoder, discriminator,
          vggnet, g_optim, e_optim, d_optim, g_ema, e_ema, device, logger=None):
    # wandb = logger
    inception = real_mean = real_cov = mean_latent = None
    if args.eval_every > 0:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()
        with open(args.inception, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]
    if get_rank() == 0:
        if args.eval_every > 0:
            with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")
        if args.log_every > 0:
            with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")

    loader = sample_data(loader)
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    d_loss_val = r1_val = 0
    loss_dict = {"d": torch.tensor(0.0, device=device), "r1": torch.tensor(0.0, device=device)}
    avg_pix_loss = util.AverageMeter()
    avg_vgg_loss = util.AverageMeter()

    if args.distributed:
        g_module = generator.module
        e_module = encoder.module
        d_module = discriminator.module if discriminator is not None else None
    else:
        g_module = generator
        e_module = encoder
        d_module = discriminator if discriminator is not None else None
    
    d_weight = torch.tensor(1.0, device=device)
    last_layer = None
    if args.use_adaptive_weight:
        if args.distributed:
            last_layer = generator.module.get_last_layer()
        else:
            last_layer = generator.get_last_layer()

    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0
    g_scale = 1
    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, args.ada_every, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_x = load_real_samples(args, loader)
    if sample_x.ndim > 4:
        sample_x = sample_x[:,0,...]

    n_step_max = max(args.n_step_d, args.n_step_e)

    requires_grad(g_ema, False)
    requires_grad(e_ema, False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        if args.debug: util.seed_everything(i)
        real_imgs = [next(loader).to(device) for _ in range(n_step_max)]

        # ========== Train Discriminator ========== #
        if args.lambda_adv > 0 and discriminator is not None:
            requires_grad(generator, False)
            requires_grad(encoder, False)
            requires_grad(discriminator, True)
            for step_index in range(args.n_step_d):
                real_img = real_imgs[step_index]
                latent_real, latent_real_logvar = encoder(real_img)
                latent_real = reparameterize(latent_real, latent_real_logvar)
                rec_img, _ = generator([latent_real], input_is_latent=True)
                if args.augment:
                    real_img_aug, _ = augment(real_img, ada_aug_p)
                    rec_img_aug, _ = augment(rec_img, ada_aug_p)
                else:
                    real_img_aug = real_img
                    rec_img_aug = rec_img
                real_pred = discriminator(real_img_aug)
                rec_pred = discriminator(rec_img_aug)
                d_loss_real = F.softplus(-real_pred).mean()
                d_loss_rec = F.softplus(rec_pred).mean()

                d_loss = d_loss_real + d_loss_rec * args.lambda_rec_d
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
                if args.augment:
                    real_img_aug, _ = augment(real_img, ada_aug_p)
                else:
                    real_img_aug = real_img
                real_pred = discriminator(real_img_aug)
                r1_loss = d_r1_loss(real_pred, real_img)
                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
                d_optim.step()
            loss_dict["r1"] = r1_loss

        # ============================ #

        # ========== Train AutoEncoder ========== #
        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        if args.debug: util.seed_everything(i)
        pix_loss = vgg_loss = adv_loss = torch.tensor(0., device=device)
        for step_index in range(args.n_step_e):
            real_img = real_imgs[step_index]
            latent_real_mean, latent_real_logvar = encoder(real_img)
            latent_real_sample = reparameterize(latent_real_mean, latent_real_logvar)
            rec_img, _ = generator([latent_real_sample], input_is_latent=True)
            if args.lambda_pix > 0:
                if args.pix_loss == 'l2':
                    pix_loss = torch.mean((rec_img - real_img) ** 2)
                elif args.pix_loss == 'l1':
                    pix_loss = F.l1_loss(rec_img, real_img)
            if args.lambda_vgg > 0:
                vgg_loss = torch.mean((vggnet(real_img) - vggnet(rec_img)) ** 2)
            if args.lambda_adv > 0:
                if args.augment:
                    rec_img_aug, _ = augment(rec_img, ada_aug_p)
                else:
                    rec_img_aug = rec_img
                rec_pred = discriminator(rec_img_aug)
                adv_loss = g_nonsaturating_loss(rec_pred)
            rec_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg
            if args.use_adaptive_weight and i >= args.disc_iter_start:
                g_loss = adv_loss * args.lambda_adv
                d_weight = calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)

            ae_loss = (
                pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg + 
                d_weight * adv_loss * args.lambda_adv
            )
            kld_loss = -0.5 * torch.sum(1 + latent_real_logvar - latent_real_mean ** 2 - latent_real_logvar.exp(), dim=1).mean()
            elbo_loss = ae_loss + kld_loss * args.lambda_kld
            loss_dict["ae"] = ae_loss
            loss_dict["rec"] = rec_loss
            loss_dict["kld"] = kld_loss
            loss_dict["pix"] = pix_loss
            loss_dict["vgg"] = vgg_loss
            loss_dict["adv"] = adv_loss

            encoder.zero_grad()
            generator.zero_grad()
            elbo_loss.backward()
            e_optim.step()
            if args.g_decay is not None:
                scale_grad(generator, g_scale)
                g_scale *= args.g_decay
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

        # ========== Update EMA ==========
        if args.ema_decay is None:
            ema_nimg = args.ema_kimg * 1000
            if args.ema_rampup is not None:
                ema_nimg = min(ema_nimg, i * args.batch * args.ema_rampup)
            accum = 0.5 ** (args.batch / max(ema_nimg, 1e-8))
        else:
            accum = args.ema_decay
        accumulate(g_ema, g_module, 0 if args.no_ema_g else accum)
        accumulate(e_ema, e_module, 0 if args.no_ema_e else accum)

        # ========== Negative Augmentation ==========
        do_nda = args.nda_every > 0 and i % args.nda_every == 0
        if args.use_nda and do_nda and i >= args.nda_iter_start:
            g_nda = g_ema
            e_nda = e_ema
            # 1. sample from prior
            noise = mixing_noise(args.batch, args.latent, 0, device)
            fake_sample, _ = g_nda(noise, input_is_latent=True)
            fake_sample = fake_sample.detach()

            # 2. minimize elbo on sampled data
            w_fake_mean, w_fake_logvar = encoder(fake_sample)
            w_fake_sample = reparameterize(w_fake_mean, w_fake_logvar)
            x_fake_rec, _ = generator([w_fake_sample], input_is_latent=True)
            if args.lambda_pix > 0:
                if args.pix_loss == 'l2':
                    pix_loss = torch.mean((x_fake_rec - fake_sample) ** 2)
                elif args.pix_loss == 'l1':
                    pix_loss = F.l1_loss(x_fake_rec, fake_sample)
            if args.lambda_vgg > 0:
                vgg_loss = torch.mean((vggnet(x_fake_rec) - vggnet(fake_sample)) ** 2)
            rec_loss = pix_loss * args.lambda_pix + vgg_loss * args.lambda_vgg
            kld_loss = -0.5 * torch.sum(1 + w_fake_logvar - w_fake_mean ** 2 - w_fake_logvar.exp(), dim=1).mean()
            elbo_fake = rec_loss + kld_loss * args.lambda_kld
            loss_nda = -elbo_fake * args.lambda_nda

            encoder.zero_grad()
            generator.zero_grad()
            loss_nda.backward()
            e_optim.step()
            g_optim.step()
            loss_dict["nda"] = loss_nda

        # ===========================================

        loss_reduced = reduce_loss_dict(loss_dict)
        # ae_loss_val = loss_reduced["ae"].mean().item()
        rec_loss_val = loss_reduced["rec"].mean().item()
        kld_loss_val = loss_reduced["kld"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()
        pix_loss_val = loss_reduced["pix"].mean().item()
        vgg_loss_val = loss_reduced["vgg"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        avg_pix_loss.update(pix_loss_val, real_img.shape[0])
        avg_vgg_loss.update(vgg_loss_val, real_img.shape[0])

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"rec: {rec_loss_val:.4f}; "
                    f"kld: {kld_loss_val:.4f}; "
                    f"pix: {pix_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; adv: {adv_loss_val:.4f}"
                )
            )

            if i % args.log_every == 0:
                with torch.no_grad():
                    # g_ema.eval()
                    # e_ema.eval()
                    # NOTE: hardcoded not to use ema
                    g_eval = g_module
                    e_eval = e_module
                    g_eval.eval()
                    e_eval.eval()
                    nrow = int(args.n_sample ** 0.5)
                    nchw = list(sample_x.shape)[1:]
                    # Reconstruction of real images
                    latent_x, _ = e_eval(sample_x)
                    rec_real, _ = g_eval([latent_x], input_is_latent=True)
                    images_recon = torch.cat((sample_x.reshape(args.n_sample//nrow, nrow, *nchw), 
                                        rec_real.reshape(args.n_sample//nrow, nrow, *nchw)), 1)
                    images_recon = images_recon.reshape(2*args.n_sample, *nchw)
                    utils.save_image(
                        images_recon,
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-recon.png"),
                        nrow=nrow,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    ref_pix_loss = torch.sum(torch.abs(sample_x - rec_real))
                    ref_vgg_loss = torch.mean((vggnet(sample_x) - vggnet(rec_real)) ** 2) if vggnet is not None else 0
                    # Fixed fake samples and reconstructions
                    images_sample, _ = g_eval([sample_z], input_is_latent=True)
                    utils.save_image(
                        images_sample,
                        os.path.join(args.log_dir, 'sample', f"{str(i).zfill(6)}-sample.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    g_eval.train()
                    e_eval.train()
                if wandb and args.wandb:
                    logger.log({
                        "real_sample": wandb.Image(sample_x, caption="Real samples"),
                        "reconstruction": wandb.Image(torch.clamp(rec_real, -1, 1), caption="Reconstruction"),
                        "random_sample": wandb.Image(torch.clamp(images_sample, -1, 1), caption="Fake samples"),
                    })

                with open(os.path.join(args.log_dir, 'log.txt'), 'a+') as f:
                    f.write(
                        (
                            f"{i:07d}; "
                            f"rec: {rec_loss_val:.4f}; "
                            f"kld: {kld_loss_val:.4f}; "
                            f"pix: {avg_pix_loss.avg:.4f}; vgg: {avg_vgg_loss.avg:.4f}; "
                            f"ref_pix: {ref_pix_loss.item():.4f}; ref_vgg: {ref_vgg_loss.item():.4f}; "
                            f"d_weight: {d_weight.item():.4f}; "
                            f"\n"
                        )
                    )

            if wandb and args.wandb:
                logger.log(
                    {
                        "rec_loss": rec_loss_val,
                        "KLD": kld_loss_val,
                    }
                )

            if args.eval_every > 0 and i % args.eval_every == 0:
                # g_ema.eval()
                # e_ema.eval()
                g_eval = g_module
                e_eval = e_module
                g_eval.eval()
                e_eval.eval()
                with torch.no_grad():
                    if args.truncation < 1:
                        mean_latent = g_eval.mean_latent(4096)
                    nll_val = fid_sa = fid_re = float('nan')
                    # NLL
                    if 'nll' in args.which_metric:
                        nll_val = evaluate_nll(e_eval, g_eval, loader2, args)
                    # Real reconstruction FID
                    if 'fid_recon' in args.which_metric:
                        features = extract_feature_from_reconstruction(
                            e_eval, g_eval, inception, args.truncation, mean_latent, loader2, args.device,
                            mode='recon', use_reparam=True,
                        ).numpy()
                        sample_mean = np.mean(features, 0)
                        sample_cov = np.cov(features, rowvar=False)
                        fid_re = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                    if 'fid_sample' in args.which_metric:
                        features = extract_feature_from_samples(
                            g_eval, inception, args.truncation, mean_latent, 64, args.n_sample_fid, args.device,
                            prior=lambda x: x,
                        ).numpy()
                        sample_mean = np.mean(features, 0)
                        sample_cov = np.cov(features, rowvar=False)
                        fid_sa = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
                with open(os.path.join(args.log_dir, 'log_fid.txt'), 'a+') as f:
                    f.write(f"{i:07d}; rec_real: {float(fid_re):.4f}; sample: {float(fid_sa):.4f}; NLL: {float(nll_val):.4f}\n")
                # g_ema.train()
                # e_ema.train()
                g_eval.train()
                e_eval.train()
                if wandb and args.wandb:
                    logger.log({"FID Real": fid_re, "FID Sample": fid_sa, "NLL": nll_val})

            if i % args.save_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
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
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
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

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--arch", type=str, default='stylegan2', help="model architectures (stylegan2 | swagan)")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--sample_cache", type=str, default=None)
    parser.add_argument("--name", type=str, help="experiment name", default='default_exp')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')
    parser.add_argument("--log_every", type=int, default=1000, help="save samples every # iters")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoints every # iters")
    parser.add_argument("--save_latest_every", type=int, default=200, help="save latest checkpoints every # iters")
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
        default=0,
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
        default=8,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--which_encoder", type=str, default='style')
    parser.add_argument("--which_latent", type=str, default='w_tied')
    parser.add_argument("--stddev_group", type=int, default=1)
    parser.add_argument("--use_wscale", action='store_true', help="whether to use `wscale` layer in idinvert encoder")
    parser.add_argument("--vgg_ckpt", type=str, default="vgg16.pth")
    parser.add_argument("--output_layer_idx", type=int, default=23)
    parser.add_argument("--lambda_vgg", type=float, default=5e-5)
    parser.add_argument("--lambda_adv", type=float, default=0)
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="recon loss on pixel (x)")
    parser.add_argument("--lambda_rec_d", type=float, default=1.0, help="d1, recon of real image")
    parser.add_argument("--lambda_kld", type=float, default=0.00025, help="kld loss")
    parser.add_argument("--pix_loss", type=str, default='l2')
    parser.add_argument("--joint", action='store_true', help="update generator with encoder")
    parser.add_argument("--inception", type=str, default=None, help="path to precomputed inception embedding")
    parser.add_argument("--eval_every", type=int, default=1000, help="interval of metric evaluation")
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument("--n_sample_fid", type=int, default=50000, help="number of the samples for calculating FID")
    parser.add_argument("--nframe_num", type=int, default=5)
    parser.add_argument("--n_step_d", type=int, default=1)
    parser.add_argument("--n_step_e", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--e_ckpt", type=str, default=None, help="path to the checkpoint of encoder")
    parser.add_argument("--g_ckpt", type=str, default=None, help="path to the checkpoint of generator")
    parser.add_argument("--d_ckpt", type=str, default=None, help="path to the checkpoint of discriminator")
    parser.add_argument("--d2_ckpt", type=str, default=None, help="path to the checkpoint of discriminator2")
    parser.add_argument("--train_from_scratch", action='store_true')
    parser.add_argument("--limit_train_batches", type=float, default=1)
    parser.add_argument("--g_decay", type=float, default=None, help="g decay factor")
    parser.add_argument("--n_mlp_g", type=int, default=8)
    parser.add_argument("--ema_kimg", type=int, default=10, help="Half-life of the exponential moving average (EMA) of generator weights.")
    parser.add_argument("--ema_rampup", type=float, default=None, help="EMA ramp-up coefficient.")
    parser.add_argument("--no_ema_e", action='store_true')
    parser.add_argument("--no_ema_g", action='store_true')
    parser.add_argument("--which_metric", type=str, nargs='*', default=['nll', 'fid_recon', 'fid_sample'])
    parser.add_argument("--use_adaptive_weight", action='store_true', help="adaptive weight borrowed from VQGAN")
    parser.add_argument("--disc_iter_start", type=int, default=20000)
    parser.add_argument("--wandb_project", type=str, default="stylevae")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument('--nll_num_samples', type=int, default=32)
    parser.add_argument('--nll_num_batches', type=int, default=-1)
    parser.add_argument("--use_nda", action='store_true')
    parser.add_argument('--nda_iter_start', type=int, default=0)
    parser.add_argument('--nda_every', type=int, default=1)
    parser.add_argument("--lambda_nda", type=float, default=0.1)

    args = parser.parse_args()
    util.seed_everything()
    args.device = device

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # args.n_mlp = 8
    args.n_latent = int(np.log2(args.size)) * 2 - 2
    args.latent = 512
    if args.which_latent == 'w_plus':
        args.latent_full = args.latent * args.n_latent
    elif args.which_latent == 'w_tied':
        args.latent_full = args.latent
    else:
        raise NotImplementedError
    
    # ===========================
    if args.use_nda:
        args.no_ema_e = False
        args.no_ema_g = False

    args.start_iter = 0
    args.iter += 1
    util.set_log_dir(args)
    util.print_args(parser, args)

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp_g, channel_multiplier=args.channel_multiplier
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

    # Define Encoder
    if args.which_encoder == 'idinvert':
        from idinvert_pytorch.models.stylegan_encoder_network import StyleGANEncoderNet
        encoder = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, use_wscale=args.use_wscale).to(device)
        e_ema = StyleGANEncoderNet(resolution=args.size, w_space_dim=args.latent,
            which_latent=args.which_latent, use_wscale=args.use_wscale).to(device)
    else:
        from model import Encoder
        encoder = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, stddev_group=args.stddev_group,
            reparameterization=True).to(device)
        e_ema = Encoder(args.size, args.latent, channel_multiplier=args.channel_multiplier,
            which_latent=args.which_latent, stddev_group=args.stddev_group,
            reparameterization=True).to(device)
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

    if args.resume:
        if args.ckpt is None:
            args.ckpt = os.path.join(args.log_dir, 'weight', f"latest.pt")
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

    elif not args.train_from_scratch:
        if args.e_ckpt is not None:
            print("load e model:", args.e_ckpt)
            e_ckpt = torch.load(args.e_ckpt, map_location=lambda storage, loc: storage)
            encoder.load_state_dict(e_ckpt["e"])
            e_ema.load_state_dict(e_ckpt["e_ema"])
            e_optim.load_state_dict(e_ckpt["e_optim"])
        if args.g_ckpt is not None:
            print("load g model:", args.g_ckpt)
            g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
            generator.load_state_dict(g_ckpt["g"])
            g_ema.load_state_dict(g_ckpt["g_ema"])
            g_optim.load_state_dict(g_ckpt["g_optim"])
        if args.d_ckpt is not None:
            print("load d model:", args.d_ckpt)
            d_ckpt = torch.load(args.d_ckpt, map_location=lambda storage, loc: storage)
            discriminator.load_state_dict(d_ckpt["d"])
            d_optim.load_state_dict(d_ckpt["d_optim"])

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

    dataset = get_image_dataset(args, args.dataset, args.path, train=True)
    if args.limit_train_batches < 1:
        indices = torch.randperm(len(dataset))[:int(args.limit_train_batches * len(dataset))]
        dataset1 = data.Subset(dataset, indices)
    else:
        dataset1 = dataset
    loader = data.DataLoader(
        dataset1,
        batch_size=args.batch,
        sampler=data_sampler(dataset1, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
    )
    # A subset of length args.n_sample_fid for FID evaluation
    loader2 = None
    if args.eval_every > 0:
        indices = torch.randperm(len(dataset))[:args.n_sample_fid]
        dataset2 = data.Subset(dataset, indices)
        loader2 = data.DataLoader(dataset2, batch_size=64, num_workers=4, shuffle=False)
        if args.sample_cache is not None:
            load_real_samples(args, sample_data(loader2))

    logger = None
    if get_rank() == 0 and wandb is not None and args.wandb:
        # wandb.init(project=args.wandb_project, name=args.name, config=args)
        logger = util.set_up_wandb(args)
    util.print_models([generator, discriminator, encoder], args)

    train(
        args, loader, loader2, generator, encoder, discriminator,
        vggnet, g_optim, e_optim, d_optim, g_ema, e_ema, device, logger,
    )

"""Fit a grouped FDN to an ambisonic spatial RIR.

The supplied ``srir_r1.wav`` is a second-order (nine-channel) ambisonic RIR.
The GFDN groups model distinct decay slopes: each group has its own learnable
GEQ absorption filter. A mono source is injected through a learnable input
gain, while a separate learnable output gain reads every GFDN group into each
ambisonic component.
"""

import argparse
import os
import time

import soundfile as sf
import torch

from e8_gfdn import (GroupedFDN, NoisyLoss, extract_background_noise,
                     plot_loss_history)
from flamo.functional import find_onset, signal_gallery
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.loss import (MultiChannelSTFTLoss, MultiResolutionSCMLoss,
                                 edc_loss, edr_loss, sparsity_loss)
from flamo.optimize.trainer import Trainer
from flamo.utils import save_audio


def load_ambisonic_rir(path: str, nfft: int, n_sh: int,
                       dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """Read, onset-align and pad an order-``n_sh`` ambisonic RIR.

    SoundFile returns audio as ``(time, channel)``; FLAMO uses
    ``(batch, time, channel)``.  The order/channel check prevents accidentally
    treating an arbitrary multichannel WAV as an ambisonic RIR.
    """
    rir, sample_rate = sf.read(path, always_2d=True)
    expected_channels = (n_sh + 1)**2
    if rir.shape[1] != expected_channels:
        raise ValueError(
            f"Order {n_sh} needs {expected_channels} channels, but {path} has "
            f"{rir.shape[1]}. Choose --n_sh to match the RIR.")
    rir = torch.as_tensor(rir, dtype=dtype)
    onset = find_onset(torch.linalg.vector_norm(rir, dim=-1))
    rir = rir[onset:onset + nfft]
    if rir.shape[0] < nfft:
        rir = torch.nn.functional.pad(rir, (0, 0, 0, nfft - rir.shape[0]))
    return (rir / rir.abs().max().clamp_min(
        torch.finfo(dtype).eps)).unsqueeze(0), sample_rate


def estimate_spatial_noise(target: torch.Tensor,
                           sample_rate: int) -> tuple[torch.Tensor, list[int]]:
    """Estimate an independent stationary noise floor for every SH channel."""
    noise, starts = [], []
    for channel in range(target.shape[-1]):
        channel_noise, start = extract_background_noise(
            target[0, :, channel], sample_rate)
        noise.append(channel_noise)
        starts.append(start)
    return torch.stack(noise, dim=-1).unsqueeze(0), starts


def example_spatial_gfdn(args) -> None:
    """Match a multichannel GFDN to an ambisonics multislope RIR"""
    target, target_fs = load_ambisonic_rir(args.target_rir, args.nfft,
                                           args.n_sh, args.dtype)
    if args.samplerate is None:
        args.samplerate = target_fs
    elif args.samplerate != target_fs:
        raise ValueError(
            f"Target sample rate is {target_fs}; use --samplerate {target_fs}."
        )

    # The GFDN itself is deterministic. Add a fixed, independently estimated
    # noise floor for each ambisonic component only when evaluating the losses.
    if args.model_noise_floor:
        noise, noise_starts = estimate_spatial_noise(target, args.samplerate)
        print("Target noise-floor start (s) per SH channel:",
              [round(start / args.samplerate, 3) for start in noise_starts])
        noise = noise.to(device=args.device, dtype=args.dtype)

    n_channels = (args.n_sh + 1)**2
    # Groups describe different spectral decay slopes, not SH components.
    # ``b`` is a learned mono-to-state vector. ``C(z)`` is a GEQ-valued
    # output matrix: every SH component has a distinct frequency-dependent
    # readout from every delay line in every slope group.
    base_delays = [661, 769, 887, 1031]
    model = GroupedFDN(
        nfft=args.nfft,
        fs=args.samplerate,
        in_ch=1,
        out_ch=n_channels,
        group_size=len(base_delays),
        n_groups=args.n_groups,
        delay_lengths=base_delays * args.n_groups,
        filter_type=args.absorption_filter_type,
        output_type=args.output_type,
        alias_decay_db=args.alias_decay_db,
        is_twostage=args.is_twostage,
        device=args.device,
        dtype=args.dtype,
    )

    with torch.no_grad():
        initial = model.get_time_response(fs=args.samplerate,
                                          identity=False).squeeze(0)
        save_audio(os.path.join(args.train_dir, "ir_init.wav"),
                   initial / initial.abs().max(), args.samplerate)

    impulse = signal_gallery(1,
                             n_samples=args.nfft,
                             n=1,
                             signal_type="impulse",
                             fs=args.samplerate,
                             device=args.device,
                             dtype=args.dtype)
    dataset = Dataset(impulse,
                      target,
                      expand=args.num,
                      device=args.device,
                      dtype=args.dtype)
    train_loader, valid_loader = load_dataset(dataset,
                                              batch_size=args.batch_size)

    trainer = Trainer(model,
                      max_epochs=args.max_epochs,
                      patience=args.patience,
                      lr=args.lr,
                      train_dir=args.train_dir,
                      device=args.device,
                      max_grad_norm=args.max_grad_norm or None)

    if args.model_noise_floor:

        # Multichannel MRSTFT
        trainer.register_criterion(NoisyLoss(MultiChannelSTFTLoss(), noise),
                                   args.stft_weight)
        # Diffuse SCM
        if args.scm_weight > 0.0:
            trainer.register_criterion(
                NoisyLoss(
                    MultiResolutionSCMLoss(sample_rate=args.samplerate,
                                           late_start_s=args.scm_late_start,
                                           block_frames=0), noise),
                args.scm_weight)

        # Energy decay based losses
        # ``edr_loss`` and ``edc_loss`` preserve the final SH-channel axis.
        if args.edr_weight > 0.0:
            trainer.register_criterion(
                NoisyLoss(
                    edr_loss(sample_rate=args.samplerate, energy_norm=True),
                    noise), args.edr_weight)
        trainer.register_criterion(
            NoisyLoss(
                edc_loss(sample_rate=args.samplerate,
                         is_broadband=True,
                         energy_norm=True,
                         convergence=True), noise), args.edc_weight)

    else:
        # Multichannel MRSTFT
        trainer.register_criterion(MultiChannelSTFTLoss(), args.stft_weight)

        # Diffuse SCM
        if args.scm_weight > 0.0:
            trainer.register_criterion(
                MultiResolutionSCMLoss(sample_rate=args.samplerate,
                                       late_start_s=args.scm_late_start,
                                       block_frames=0), args.scm_weight)

        # Energy decay based losses
        # ``edr_loss`` and ``edc_loss`` preserve the final SH-channel axis.
        if args.edr_weight > 0.0:
            trainer.register_criterion(
                edr_loss(sample_rate=args.samplerate, energy_norm=True),
                args.edr_weight)
        trainer.register_criterion(
            edc_loss(sample_rate=args.samplerate,
                     is_broadband=True,
                     energy_norm=True,
                     convergence=True), args.edc_weight)

    trainer.register_criterion(sparsity_loss(),
                               args.sparsity_weight,
                               requires_model=True)

    trainer.train(train_loader, valid_loader)
    plot_loss_history(trainer, args.train_dir)

    with torch.no_grad():
        optimized = model.get_time_response(fs=args.samplerate,
                                            identity=False).squeeze(0)
        save_audio(os.path.join(args.train_dir, "ir_optim.wav"),
                   optimized / optimized.abs().max(),
                   args.samplerate,
                   subtype="FLOAT")
        if args.model_noise_floor:
            optimized_noisy = optimized + noise.squeeze(0)
            save_audio(os.path.join(args.train_dir, "ir_optim_noisy.wav"),
                       optimized_noisy / optimized_noisy.abs().max(),
                       args.samplerate,
                       subtype="FLOAT")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_rir", default="rirs/multi-slope/srir_r1.wav")
    parser.add_argument("--n_sh",
                        type=int,
                        default=2,
                        help="ambisonics order; channel count is (N_sh + 1)^2")
    parser.add_argument("--nfft", type=int, default=96000)
    parser.add_argument("--samplerate",
                        type=int,
                        default=32000,
                        help="defaults to the target RIR sample rate")
    parser.add_argument("--model_noise_floor",
                        action="store_true",
                        help="whether to model noise floor")
    parser.add_argument("--dtype",
                        choices=["float32", "float64"],
                        default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="clip the global gradient norm; use 0 to disable")
    parser.add_argument("--n_groups",
                        type=int,
                        default=2,
                        help="number of independently filtered GFDN groups")
    parser.add_argument("--is_twostage", action="store_true")
    parser.add_argument("--alias_decay_db", type=float, default=30.0)
    parser.add_argument("--stft_weight", type=float, default=1.0)
    parser.add_argument("--edr_weight", type=float, default=0.0)
    parser.add_argument("--edc_weight", type=float, default=1.0)
    parser.add_argument("--scm_weight",
                        type=float,
                        default=0.0,
                        help="weight of the late-tail spatial covariance loss")
    parser.add_argument("--scm_late_start",
                        type=float,
                        default=0.08,
                        help="late-tail SCM analysis start time in seconds")
    parser.add_argument("--sparsity_weight", type=float, default=0.01)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument(
        "--absorption-filter_type",
        type=str,
        default='geq',
        choices=['scalar', 'shelf', 'peq', 'geq'],
        help='choice of attenuation filter',
    )

    parser.add_argument(
        "--output_type",
        type=str,
        default='scalar',
        choices=['scalar', 'geq'],
        help='choice of output weights',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.train_dir is None:
        args.train_dir = os.path.join(
            "output", "spatial_gfdn_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.train_dir, exist_ok=True)
    with open(os.path.join(args.train_dir, "args.txt"), "w") as handle:
        handle.write("\n".join(f"{key},{value}"
                               for key, value in sorted(vars(args).items())))
    example_spatial_gfdn(args)

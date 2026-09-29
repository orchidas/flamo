from typing import Optional
import torch
import torch.nn as nn
import argparse
import numpy as np
import os
import time
import auraloss
import pyrato
import pyfar as pf
import soundfile as sf
import matplotlib.pyplot as plt

from collections import OrderedDict

from flamo.auxiliary.reverb import parallelGFDNFirstOrderShelving, parallelGFDNPEQ, parallelGFDNGEQ, parallelGFDNScalar
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.loss import sparsity_loss, edr_loss, edc_loss
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.utils import save_audio
from flamo.functional import signal_gallery, find_onset

torch.manual_seed(130798)


def plot_loss_history(trainer: Trainer, output_dir: str) -> None:
    """Save total training and validation loss against epoch number."""
    epochs = range(1, len(trainer.train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, trainer.train_loss, marker="o", label="Training loss")
    ax.plot(epochs, trainer.valid_loss, marker="o", label="Validation loss")
    ax.set(
        xlabel="Epoch",
        ylabel="Total loss",
        title="GFDN training loss",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
    plt.close(fig)


def extend_noise(noise: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Extend a noise segment to ``n_samples`` via phase randomization: the
    segment's magnitude spectrum is interpolated onto the frequency grid of
    the target length and combined with uniformly random phases. The result
    is rescaled to the segment's mean power.

    Args:
        noise: 1D noise segment.
        n_samples: length of the extended noise.
    Returns:
        1D array of length ``n_samples``.
    """
    mag = np.abs(np.fft.rfft(noise))
    freqs = np.fft.rfftfreq(len(noise))
    new_freqs = np.fft.rfftfreq(n_samples)
    new_mag = np.interp(new_freqs, freqs, mag)
    phase = np.random.uniform(-np.pi, np.pi, len(new_freqs))
    phase[0] = 0
    if n_samples % 2 == 0:
        phase[-1] = 0  # Nyquist bin must be real
    out = np.fft.irfft(new_mag * np.exp(1j * phase), n=n_samples)
    return out * np.sqrt(np.mean(noise**2) / np.mean(out**2))


def extract_background_noise(rir: torch.Tensor,
                             fs: int) -> tuple[torch.Tensor, int]:
    """
    Estimate the background noise of a RIR: the intersection time between
    the decay and the noise floor is detected with pyrato's Lundeby method,
    the RIR tail from that point on is taken as the noise segment, and the
    segment is extended to the full RIR length via :func:`extend_noise`.

    Args:
        rir: 1D RIR tensor, onset-aligned (no leading silence).
        fs: sample rate in Hz.
    Returns:
        ``(noise, start)`` where ``noise`` is a 1D tensor with the same length,
        dtype and device as ``rir``, and ``start`` is the sample index where
        the noise segment begins.
    """
    x = rir.detach().cpu().numpy().astype(np.float64).flatten()
    intersection_time = pyrato.intersection_time_lundeby(pf.Signal(x, fs),
                                                         freq="broadband")[0]
    start = int(np.round(np.squeeze(intersection_time) * fs))
    if start >= int(len(x) * 0.99):
        start = int(len(x) *
                    0.99)  # noise floor beyond the RIR: take the last 1%
    noise = extend_noise(x[start:], len(x))
    return torch.as_tensor(noise, dtype=rir.dtype, device=rir.device), start


class NoisyLoss(nn.Module):
    """
    Wrap a criterion so that a fixed noise term is summed to the model output
    before the loss is computed. This lets a noise-free model (e.g. a GFDN)
    be fit to a target RIR with a background noise floor, without the loss
    penalizing the missing noise in the tail of the EDC/EDR.

    Args:
        criterion: the loss to wrap, called as ``criterion(y_pred, y_true)``.
        noise: noise term of shape ``(1, n_samples, n_channels)``, broadcast
            over the batch.
    """

    def __init__(self, criterion: nn.Module, noise: torch.Tensor):
        super().__init__()
        self.criterion = criterion
        self.register_buffer("noise", noise)

    def forward(self, y_pred, y_true):
        return self.criterion(y_pred + self.noise.to(y_pred.dtype), y_true)


class MultiResoSTFT(nn.Module):
    """compute the mean absolute error between the auraloss of two RIRs"""

    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, rir1, rir2):
        return self.MRstft(rir1.permute(0, 2, 1), rir2.permute(0, 2, 1))


class GroupedFDN(system.Shell):
    """
    Grouped Feedback Delay Network (FDN), after Das & Abel [1]_.

    The ``n_groups * group_size`` delay lines are split into ``n_groups``
    groups. Within a group the lines are mixed by an orthogonal matrix built
    from that group's entry in ``mixing_angles``; the groups are then coupled
    to one another according to ``coupling_angles``. Delay lengths and the
    feedback (mixing + coupling) matrix are fixed. Only the input/output
    gains and the two parameters of the shared first-order shelving
    attenuation filter (the DC-frequency reverberation time and the shelving
    crossover) are learnable.

    ``group_size`` must be a power of two: the intra-group mixing matrix is
    built by repeated Kronecker self-products of a 2x2 rotation.

    .. [1] Das, O., & Abel, J. S. (2021). Grouped feedback delay networks for
       modeling of coupled spaces. J. Audio Eng. Soc, 69(7/8), 486-496.
    """

    def __init__(
        self,
        nfft: int,
        fs: int,
        in_ch: int,
        out_ch: int,
        group_size: int,
        n_groups: int,
        delay_lengths: torch.Tensor | list[int],
        filter_type: str = 'peq',
        rt_dc: Optional[float] = 1.0,
        rt_nyquist: Optional[float] = 0.2,
        crossover_freq: Optional[float] = 4000.0,
        is_twostage: bool = False,
        alias_decay_db: Optional[float] = 0.0,
        device: Optional[str] = 'cuda',
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        assert group_size >= 2 and group_size & (group_size - 1) == 0, (
            f"group_size must be a power of two >= 2, got {group_size}")

        n_delays = group_size * n_groups
        delay_lengths = torch.as_tensor(delay_lengths,
                                        device=device,
                                        dtype=torch.int64)
        assert delay_lengths.numel() == n_delays, (
            f"delay_lengths must have {n_delays} entries (group_size * n_groups), got {delay_lengths.numel()}"
        )

        input_gain = dsp.Gain(
            size=(n_delays, in_ch),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        output_gain = dsp.Gain(
            size=(out_ch, n_delays),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        delays = dsp.parallelDelay(
            size=(n_delays, ),
            max_len=delay_lengths.max(),
            nfft=nfft,
            isint=True,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )
        delays.assign_value(delays.sample2s(delay_lengths))

        mixing_matrix = dsp.Matrix(
            size=(n_delays, n_delays),
            nfft=nfft,
            matrix_type="random_block_diagonal",
            n_blocks=n_groups,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
            dtype=dtype,
        )

        if filter_type == 'scalar':
            attenuation = parallelGFDNScalar(n_groups=n_groups,
                                             delays=delay_lengths,
                                             fs=fs,
                                             nfft=nfft,
                                             alias_decay_db=alias_decay_db,
                                             requires_grad=True,
                                             device=device,
                                             dtype=dtype)

        elif filter_type == 'shelf':
            attenuation = parallelGFDNFirstOrderShelving(
                nfft=nfft,
                fs=fs,
                rt_nyquist=rt_nyquist,
                n_groups=n_groups,
                delays=delay_lengths,
                alias_decay_db=alias_decay_db,
                requires_grad=True,
                device=device,
                dtype=dtype,
            )
            omega_c = 2 * torch.pi * crossover_freq / fs
            attenuation.assign_value(
                torch.tensor([rt_dc, omega_c], device=device, dtype=dtype))

        elif filter_type == 'peq':
            attenuation = parallelGFDNPEQ(
                n_bands=8,
                f_min=20,
                f_max=2000,
                nfft=nfft,
                fs=fs,
                delays=delay_lengths,
                design='biquad',
                is_twostage=is_twostage,
                is_proportional=False,
                n_groups=n_groups,
                alias_decay_db=alias_decay_db,
                requires_grad=True,
                device=device,
                dtype=dtype,
            )

        elif filter_type == 'geq':
            attenuation = parallelGFDNGEQ(octave_interval=1,
                                          n_groups=n_groups,
                                          nfft=nfft,
                                          fs=fs,
                                          delays=delay_lengths,
                                          alias_decay_db=alias_decay_db,
                                          requires_grad=True,
                                          device=device,
                                          dtype=dtype)
        else:
            raise ValueError('Filter type must be shelf / peq / geq')

        feedback = system.Series(
            OrderedDict({
                "mixing_matrix": mixing_matrix,
                "attenuation": attenuation
            }))
        feedback_loop = system.Recursion(fF=delays, fB=feedback)

        core = system.Series(
            OrderedDict({
                "input_gain": input_gain,
                "feedback_loop": feedback_loop,
                "output_gain": output_gain,
            }))

        input_layer = dsp.FFT(nfft, dtype=dtype)
        output_layer = dsp.iFFTAntiAlias(nfft=nfft,
                                         alias_decay_db=alias_decay_db,
                                         device=device,
                                         dtype=dtype)
        super().__init__(core=core,
                         input_layer=input_layer,
                         output_layer=output_layer)


def example_gfdn(args):
    """
    Example function that demonstrates the construction and training of a
    Grouped Feedback Delay Network (GFDN) model via simple gradient descent
    directly on the GFDN's own parameters (input/output gains and attenuation
    filter), analogous to `example_fdn` in e8_fdn.py.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """

    # read target
    target_rir = torch.tensor(sf.read(args.target_rir)[0], dtype=torch.float32)
    target_rir = target_rir / torch.max(torch.abs(target_rir))
    rir_onset = find_onset(target_rir)
    target_rir = target_rir[rir_onset:(rir_onset + args.nfft)].view(1, -1, 1)
    # zero pad to nfft
    target_rir = torch.nn.functional.pad(
        target_rir, (0, 0, 0, args.nfft - target_rir.shape[1]))

    # background noise of the target, summed to the GFDN output in the loss
    # (zero-padded like the target)
    target_rir_1d = target_rir.flatten()
    noise, noise_start = extract_background_noise(target_rir_1d,
                                                  args.samplerate)
    print(
        f"Target RIR noise floor starts at {noise_start / args.samplerate:.3f} s"
    )
    noise = torch.nn.functional.pad(noise, (0, args.nfft - noise.shape[0]))
    noise = noise.view(1, -1, 1).to(device=args.device, dtype=args.dtype)

    # GFDN parameters
    delays = [997, 1153, 1327, 1559]
    group_size = len(delays)
    n_groups = 2
    alias_decay_db = 30

    ## ---------------- CONSTRUCT GFDN ---------------- ##

    model = GroupedFDN(
        group_size=group_size,
        n_groups=n_groups,
        delay_lengths=delays * n_groups,
        nfft=args.nfft,
        fs=args.samplerate,
        in_ch=1,
        out_ch=1,
        rt_dc=1.0,
        rt_nyquist=0.2,
        crossover_freq=4000.0,
        alias_decay_db=alias_decay_db,
        is_twostage=args.is_twostage,
        filter_type=args.filter_type,
        device=args.device,
        dtype=args.dtype,
    )

    # Get initial impulse response
    with torch.no_grad():
        ir_init = model.get_time_response(identity=False,
                                          fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_init.wav"),
            ir_init / torch.max(torch.abs(ir_init)),
            fs=args.samplerate,
        )

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    # read target RIR
    input = signal_gallery(
        1,
        n_samples=args.nfft,
        n=1,
        signal_type="impulse",
        fs=args.samplerate,
        device=args.device,
        dtype=args.dtype,
    )

    dataset = Dataset(
        input=input,
        target=target_rir,
        expand=args.num,
        device=args.device,
        dtype=args.dtype,
    )
    train_loader, valid_loader = load_dataset(dataset,
                                              batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(
        model,
        max_epochs=args.max_epochs,
        patience=20,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device,
    )
    trainer.register_criterion(NoisyLoss(MultiResoSTFT(), noise), 1)
    trainer.register_criterion(sparsity_loss(), 1, requires_model=True)
    # trainer.register_criterion(NoisyLoss(edr_loss(sample_rate=args.samplerate), noise), 1)
    # trainer.register_criterion(
    #     NoisyLoss(
    #         edc_loss(sample_rate=args.samplerate,
    #                  is_broadband=True,
    #                  convergence=True), noise), 1)

    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)
    plot_loss_history(trainer, args.train_dir)

    # Get optimized impulse response
    with torch.no_grad():
        ir_optim = model.get_time_response(identity=False,
                                           fs=args.samplerate).squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim.wav"),
            ir_optim / torch.max(torch.abs(ir_optim)),
            fs=args.samplerate,
        )

        ir_optim_noisy = ir_optim + noise.squeeze()
        save_audio(
            os.path.join(args.train_dir, "ir_optim_noisy.wav"),
            ir_optim_noisy / torch.max(torch.abs(ir_optim_noisy)),
            fs=args.samplerate,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=96000, help="FFT size")
    parser.add_argument("--samplerate",
                        type=int,
                        default=48000,
                        help="sampling rate")
    parser.add_argument("--dtype",
                        type=str,
                        default="float64",
                        choices=["float32", "float64"],
                        help="data type for tensors")
    parser.add_argument("--num", type=int, default=100, help="dataset size")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="device to use for computation")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="batch size for training")
    parser.add_argument("--max_epochs",
                        type=int,
                        default=20,
                        help="maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--train_dir",
                        type=str,
                        help="directory to save training results")
    parser.add_argument("--masked_loss",
                        type=bool,
                        default=False,
                        help="use masked loss")
    parser.add_argument("--is_twostage",
                        action="store_true",
                        help="use two stage attenuation filter")
    parser.add_argument(
        "--filter_type",
        type=str,
        default='peq',
        choices=['scalar', 'shelf', 'peq', 'geq'],
        help='choice of attenuation filter',
    )
    parser.add_argument(
        "--target_rir",
        type=str,
        default="rirs/multi-slope/ertd1_rir_r1.wav",
        help="filepath to target RIR",
    )

    args = parser.parse_args()

    # Check for compatible device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # convert dtype string to torch dtype
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # Save arguments
    with open(os.path.join(args.train_dir, "args.txt"), "w") as f:
        f.write("\n".join([
            str(k) + "," + str(v)
            for k, v in sorted(vars(args).items(), key=lambda x: x[0])
        ]))

    # Run the example GFDN training
    example_gfdn(args)

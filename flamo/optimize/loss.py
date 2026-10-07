import torch
import torch.nn as nn
import numpy as np
import auraloss
from flamo.optimize.utils import generate_partitions
from flamo.processor.dsp import HouseholderMatrix
from nnAudio import features
import pyfar as pf
import torch.nn.functional as F
from typing import List


def _finite_audio(x: torch.Tensor, max_amplitude: float = 1e6) -> torch.Tensor:
    """Replace non-finite samples before an energy/logarithm calculation.

    A temporarily unstable recursive filter can otherwise make the EDC
    normalization evaluate ``inf / inf``.  The clamp is far above normal RIR
    amplitudes and has no effect during regular optimisation.
    """
    return torch.nan_to_num(x,
                            nan=0.0,
                            posinf=max_amplitude,
                            neginf=-max_amplitude).clamp(
                                -max_amplitude, max_amplitude)


class MultiChannelSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for signals shaped ``(batch, time, channel)``.

    Unlike losses which flatten the channel axis before calculating an STFT,
    this criterion keeps each channel independent and averages their spectral
    convergence and log-magnitude errors.  It is consequently suitable for
    ambisonic impulse responses, where channels must not be summed before the
    analysis.
    """

    def __init__(self,
                 fft_sizes: List[int] = (512, 1024, 2048),
                 overlap: float = 0.75,
                 log_epsilon: float = 1e-7):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.overlap = overlap
        self.log_epsilon = log_epsilon
        # auraloss calculates spectral-convergence and log-magnitude terms at
        # each FFT resolution. Its expected layout is (batch, channel, time),
        # which preserves, rather than sums, ambisonic components.
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=list(self.fft_sizes),
            hop_sizes=[
                max(1, int(size * (1 - overlap))) for size in self.fft_sizes
            ],
            win_lengths=list(self.fft_sizes),
        )

    @staticmethod
    def _check_inputs(y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.ndim != 3 or y_pred.shape != y_true.shape:
            raise ValueError(
                "Expected equal tensors of shape (batch, time, channel).")

    @staticmethod
    def _stft(x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        # torch.stft accepts at most two dimensions. Flatten batch/channel
        # into the batch axis, then restore the channel-preserving layout.
        batch_size, n_samples, n_channels = x.shape
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        stft = torch.stft(x.permute(0, 2, 1).reshape(-1, n_samples),
                          n_fft=n_fft,
                          hop_length=hop_length,
                          window=window,
                          return_complex=True,
                          center=True)
        return stft.reshape(batch_size, n_channels, *stft.shape[-2:]).abs()

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        self._check_inputs(y_pred, y_true)
        return self.mrstft(
            _finite_audio(y_pred).transpose(1, 2),
            _finite_audio(y_true).transpose(1, 2))


class MultiResolutionSCMLoss(nn.Module):
    """Late-tail spatial covariance loss for multichannel impulse responses.

    A complex spatial covariance matrix (SCM) is calculated independently for
    each STFT frequency bin and late-time frame region. Trace normalization
    makes the criterion sensitive to interchannel power and coherence, while
    remaining independent of the overall decay energy; pair it with an EDC or
    EDR loss to constrain that energy decay.

    Inputs have shape ``(batch, time, channel)``. The channel ordering and
    normalization are intentionally untouched, so matching AmbiX ACN/SN3D
    target and prediction tensors can be used directly.
    """

    def __init__(self,
                 fft_sizes: List[int] = (512, 1024, 2048),
                 overlap: float = 0.75,
                 sample_rate: int = 48000,
                 late_start_s: float = 0.08,
                 block_frames: int = 0,
                 epsilon: float = 1e-8):
        super().__init__()
        if not 0 <= late_start_s:
            raise ValueError("late_start_s must be non-negative")
        if block_frames < 0:
            raise ValueError(
                "block_frames must be zero (whole tail) or positive")
        self.fft_sizes = tuple(fft_sizes)
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.late_start_s = late_start_s
        self.block_frames = block_frames
        self.epsilon = epsilon

    @staticmethod
    def _check_inputs(y_pred: torch.Tensor, y_true: torch.Tensor):
        if y_pred.ndim != 3 or y_pred.shape != y_true.shape:
            raise ValueError(
                "Expected equal tensors of shape (batch, time, channel).")

    @staticmethod
    def _complex_stft(x: torch.Tensor, n_fft: int,
                      hop_length: int) -> torch.Tensor:
        batch_size, n_samples, n_channels = x.shape
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        # Restore (batch, channel, frequency, frame) after flattening the
        # batch/channel axes required by torch.stft.
        stft = torch.stft(x.permute(0, 2, 1).reshape(-1, n_samples),
                          n_fft=n_fft,
                          hop_length=hop_length,
                          window=window,
                          return_complex=True,
                          center=True)
        return stft.reshape(batch_size, n_channels, *stft.shape[-2:])

    def _scm(self, x: torch.Tensor, n_fft: int) -> torch.Tensor | None:
        hop_length = max(1, int(n_fft * (1 - self.overlap)))
        stft = self._complex_stft(_finite_audio(x), n_fft, hop_length)
        start_frame = int(
            np.ceil(self.late_start_s * self.sample_rate / hop_length))
        stft = stft[..., start_frame:]
        n_late_frames = stft.shape[-1]
        if n_late_frames == 0:
            return None
        if self.block_frames == 0:
            # Standard SCM estimate: average every frame in the selected late
            # region into one covariance matrix per STFT frequency bin.
            stft = stft.unsqueeze(-2)
            frames_per_scm = n_late_frames
        else:
            n_blocks = n_late_frames // self.block_frames
            if n_blocks == 0:
                return None
            stft = stft[..., :n_blocks * self.block_frames]
            stft = stft.reshape(*stft.shape[:-1], n_blocks, self.block_frames)
            frames_per_scm = self.block_frames
        # stft dimes are (batch, channel, frequency, 1, num_frames) if block_frames = 0
        # else (batch, channel, frequency, SCM block, frame-within-block)
        # scm size is (batch, frequency, SCM block, channel, channel)
        scm = torch.einsum("bcfkl,bdfkl->bfkcd", stft, stft.conj())
        scm = scm / frames_per_scm
        trace = scm.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)
        # Trace normalization retains spatial power distribution and complex
        # interchannel coherence but delegates total late energy to the EDC.
        return scm / trace.clamp_min(self.epsilon)[..., None, None]

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        self._check_inputs(y_pred, y_true)
        loss = y_pred.new_zeros(())
        n_used = 0
        for n_fft in self.fft_sizes:
            if n_fft > y_pred.shape[1]:
                continue
            pred_scm = self._scm(y_pred, n_fft)
            true_scm = self._scm(y_true, n_fft)
            if pred_scm is None or true_scm is None:
                continue
            loss = loss + (pred_scm - true_scm).abs().square().mean()
            n_used += 1
        if n_used == 0:
            raise ValueError("The late region contains no complete SCM block.")
        return loss / n_used


# wrapper for the sparsity loss
class sparsity_loss(nn.Module):
    r"""
    Calculates the sparsity loss for a given model.

    The sparsity loss is calculated based on the feedback loop of the FDN model's core.
    It measures the sparsity of the feedback matrix A of size (N, N).
    Note: for the loss to be compatible with the class :class:`flamo.optimize.trainer.Trainer`, it requires :attr:`y_pred` and :attr:`y_target` as arguments even if these are not being considered.
    If the feedback matrix has a third dimension C, A.size = (C, N, N), the loss is calculated as the mean of the contribution of each (N,N) matrix.

    .. math::

        \mathcal{L} = \frac{\sum_{i,j} |A_{i,j}| - N\sqrt{N}}{N(1 - \sqrt{N})}

    For more details, refer to the paper `Optimizing Tiny Colorless Feedback Delay Networks <https://arxiv.org/abs/2402.11216>`_ by Dal Santo, G. et al.

    **Arguments**:
        - **y_pred** (torch.Tensor): The predicted output.
        - **y_target** (torch.Tensor): The target output.
        - **model** (nn.Module): The model containing the core with the feedback loop.

    Returns:
        torch.Tensor: The calculated sparsity loss.
    """

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor,
                model: nn.Module):
        core = model.get_core()
        # Try to get the mixing matrix from different possible locations
        mixing_matrix = None
        try:
            mixing_matrix = core.feedback_loop.feedback
            A = mixing_matrix.map(mixing_matrix.param)
        except AttributeError:
            try:
                mixing_matrix = core.feedback_loop.feedback.mixing_matrix
                A = mixing_matrix.map(mixing_matrix.param)
            except AttributeError:
                mixing_matrix = core.branchA.feedback_loop.feedback.mixing_matrix
                A = mixing_matrix.map(mixing_matrix.param)

        if isinstance(mixing_matrix, HouseholderMatrix):
            # u carries a leading batch dimension (u.shape == (batch, N, 1)); this
            # loss assumes a single shared mixing matrix, so drop it before rebuilding
            # the Householder matrix (u.shape[0] would otherwise be batch, not N).
            u = A.squeeze(0)
            A = torch.eye(u.shape[0], device=u.device,
                          dtype=u.dtype) - 2 * u @ u.T

        N = A.shape[-1]
        if mixing_matrix.matrix_type == "random_block_diagonal":
            n_blocks = mixing_matrix.n_blocks
            block_size = N // n_blocks
            loss = 0
            for i in torch.arange(0, N, block_size):
                block = A[..., i:i + block_size, i:i + block_size]
                N = block.shape[-1]
                loss += (torch.sum(torch.abs(block)) -
                         N * np.sqrt(N)) / (N * (1 - np.sqrt(N)))
            return loss

        if len(A.shape) == 3:
            return torch.mean(
                (torch.sum(torch.abs(A), dim=(-2, -1)) - N * np.sqrt(N)) /
                (N * (1 - np.sqrt(N))))

        # A = torch.matrix_exp(skew_matrix(A))
        return -(torch.sum(torch.abs(A)) - N * np.sqrt(N)) / (N *
                                                              (np.sqrt(N) - 1))


class mse_loss(nn.Module):
    r"""
    Wrapper for the mean squared error loss.

    .. math::

        \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( y_{\text{pred},i} -  y_{\text{true},i} \right)^2

    where :math:`N` is the number of nfft points and :math:`M` is the number of channels.

    **Arguments / Attributes**:
        - **nfft** (int): Number of FFT points.
        - **device** (str): Device to run the calculations on.

    """

    def __init__(self, nfft: int = None, device: str = "cpu"):

        super().__init__()
        self.nfft = nfft
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.name = "MSE"

    def forward(self, y_pred, y_true):
        """
        Calculates the mean squared error loss.
        If :attr:`is_masked` is set to True, the loss is calculated using a masked version of the predicted output. This option is useful to introduce stochasticity, as the mask is generated randomly.

        **Arguments**:
            - **y_pred** (torch.Tensor): The predicted output.
            - **y_true** (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The calculated MSE loss.
        """
        y_pred_sum = torch.sum(y_pred, dim=-1)
        return self.mse_loss(y_pred_sum, y_true.squeeze(-1))


class masked_mse_loss(nn.Module):
    r"""
    Wrapper for the mean squared error loss with random masking.

    Calculates the mean squared error loss between the predicted and target outputs.
    The loss is calculated using a masked version of the predicted output. This option is useful to introduce stochasticity, as the mask is generated randomly.

    .. math::

        \mathcal{L} = \frac{1}{\left| \mathbb{S} \right|} \sum_{i \in \mathbb{S}} \left( y_{\text{pred}, i} - y_{\text{true},i} \right)^2

    where :math:`\mathbb{S}` is the set of indices of the mask being analyzed during the training step.

    **Arguments / Attributes**:
        - **nfft** (int): Number of FFT points.
        - **n_samples** (int): Number of samples for masking.
        - **n_sets** (int): Number of sets for masking. Default is 1.
        - **regenerate_mask** (bool): After all sets are used, if True, the mask is regenerated. Default is True.
        - **device** (str): Device to run the calculations on. Default is 'cpu'.
    """

    def __init__(
        self,
        nfft: int,
        n_samples: int,
        n_sets: int = 1,
        regenerate_mask: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.n_samples = n_samples
        self.n_sets = n_sets
        self.nfft = nfft
        self.regenerate_mask = regenerate_mask
        self.mask_indices = generate_partitions(
            torch.arange(self.nfft // 2 + 1), n_samples, n_sets)
        self.i = -1

    def forward(self, y_pred, y_true):
        """
        Calculates the masked mean squared error loss.

        **Arguments**:
            - **y_pred** (torch.Tensor): The predicted output.
            - **y_true** (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The calculated masked MSE loss.
        """
        self.i += 1
        # generate random mask for sparse sampling
        if self.i >= self.mask_indices.shape[0]:
            self.i = 0
            if self.regenerate_mask:
                # generate a new mask
                self.mask_indices = generate_partitions(
                    torch.arange(self.nfft // 2 + 1), self.n_samples,
                    self.n_sets)
        mask = self.mask_indices[self.i]
        return torch.mean(torch.pow(y_pred[:, mask] - y_true[:, mask], 2))


class mel_mss_loss(nn.Module):
    r"""
    Multi-Scale Spectral Loss in the Mel scale.
    This loss function computes the difference between predicted and true audio signals
    in the Mel spectrogram domain across multiple FFT sizes.
    The number of Mel bins is determined by the current FFT size divided by 8.
    It is possible to apply a mask based on Signal-to-Noise Ratio (SNR) to the loss computation (this is particularly useful for training models on noisy data).
    The mask is calculated from the target (true) signal and is applied to both target and prediction before loss computation.
    The noise will be calculated as the mean of the energy of the last 0.01s unless its value is being passed as an argument.

    The loss is computed as the p norm of the difference between the predicted and true Mel spectrograms.
    The spectrogram is computed using nnAudio's MelSpectrogram class.

    Attributes:
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
        - **name** (str): A name for the loss function. Default is "MelMSS".
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **apply_mask** (bool): Whether to apply a mask based on SNR. Default is False.
        - **threshold** (float): The SNR threshold for masking. Default is 5.
        - **p** (str): The order of the norm to be used. Default is "fro" (Frobenius norm).
        - **log_term** (bool): Whether to include the log term in the loss computation. Default is False.
        - **alpha** (float): A scaling factor for the log term in the loss computation. Default is 1.0.
        - **noise_energy** (float): The energy of the noise to be used for masking. Default is None.
    """

    def __init__(
        self,
        nfft: List[int] = [128, 256, 512, 1024, 2048, 4096],
        overlap: float = 0.75,
        sample_rate: int = 48000,
        energy_norm: bool = False,
        device="cpu",
        name: str = "MelMSS",
        apply_mask: bool = False,
        threshold: float = 5,
        p: str = "fro",
        log_term: bool = False,
        alpha: float = 1.0,
        noise_energy=None,
    ):
        super().__init__()
        self.nfft = nfft
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.energy_norm = energy_norm
        self.name = name
        self.device = device
        self.apply_mask = apply_mask
        self.threshold = threshold
        self.p = p
        self.log_term = log_term
        self.alpha = alpha
        self.noise_energy = noise_energy

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        n_channels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]

        if self.energy_norm:
            y_pred = y_pred / torch.norm(y_pred, p=2)
            y_true = y_true / torch.norm(y_true, p=2)

        # reshape it to (num_audio, len_audio) as indicated by nnAudio
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
        y_true = torch.reshape(y_true, (-1, y_true.shape[1]))

        loss = 0  # initialize loss
        for i, nfft in enumerate(self.nfft):
            # initialize stft function with new nfft
            hop_length = int(nfft * (1 - self.overlap))
            mel_stft = features.mel.MelSpectrogram(
                n_fft=nfft,
                hop_length=hop_length,
                window="hann",
                sr=self.sample_rate,
                fmin=0,
                fmax=self.sample_rate // 2,
                n_mels=nfft // 8,
                verbose=False,
            )
            mel_stft = mel_stft.to(self.device).to(y_pred.dtype)

            h, w = tuple(mel_stft(y_pred).shape[-2:])
            Y_pred_lin = torch.reshape(mel_stft(y_pred),
                                       (batch_size, h, w, n_channels))
            Y_true_lin = torch.reshape(mel_stft(y_true),
                                       (batch_size, h, w, n_channels))

            mask = torch.ones_like(Y_true_lin)
            if self.apply_mask:
                if not self.noise_energy:
                    # compute the noise energy as the mean of the last 0.01s
                    self.noise_energy = torch.mean(
                        torch.pow(
                            Y_true_lin[:, :, -int(0.01 * self.sample_rate /
                                                  hop_length), :],
                            2,
                        ))
                SNR = 10 * torch.log10(
                    torch.max(Y_true_lin**2, self.noise_energy * 1.01) -
                    self.noise_energy) - 10 * torch.log10(self.noise_energy)
                mask[SNR < self.threshold] = 0
                N = torch.sum(mask)
            else:
                N = torch.numel(Y_true_lin)

            # update match loss
            loss += torch.norm((Y_true_lin - Y_pred_lin) * mask, p=self.p) / N
            if self.log_term:
                Y_pred_log = torch.reshape(torch.log(mel_stft(y_pred)),
                                           (batch_size, h, w, n_channels))
                Y_true_log = torch.reshape(torch.log(mel_stft(y_true)),
                                           (batch_size, h, w, n_channels))
                loss += self.alpha * torch.norm(
                    (Y_true_log - Y_pred_log) * mask, p=self.p) / N

        return loss


class mss_loss(nn.Module):
    r"""
    Multi-Scale Spectral Loss in the linear scale.
    This loss function computes the difference between predicted and true audio signals
    in the linear spectrogram domain across multiple FFT sizes.
    It is possible to apply a mask based on Signal-to-Noise Ratio (SNR) to the loss computation (this is particularly useful for training models on noisy data).
    The mask is calculated from the target (true) signal and is applied to both target and prediction before loss computation.
    The noise will be calculated as the mean of the energy of the last 0.01s unless its value is being passed as an argument.

    The loss is computed as the p norm of the difference between the predicted and true spectrograms.
    The spectrogram is computed using nnAudio's STFT class.

    Using the :arg:`form` argument, the loss can be computed in different ways:
    - **None**: The loss is computed as the p norm of the difference between the predicted and true spectrograms.
    - **yamamoto**: The loss is computed as the Frobenius norm of the difference between the predicted and true spectrograms, divided by the Frobenius norm of the true spectrogram. The log term is computed as the L1 norm of the difference between the predicted and true log spectrograms, divided by the number of elements in the true log spectrogram.
    - **magenta**: The loss is computed as the L1 norm of the difference between the predicted and true spectrograms, divided by the number of elements in the true spectrogram. The log term is computed as the L1 norm of the difference between the predicted and true log spectrograms, divided by the number of elements in the true log spectrogram.

    Attributes:
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
        - **name** (str): A name for the loss function. Default is "MelMSS".
        - **nfft** (list): A list of FFT sizes to compute the multi-scale spectrograms.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.75.
        - **apply_mask** (bool): Whether to apply a mask based on SNR. Default is False.
        - **threshold** (float): The SNR threshold for masking. Default is 5.
        - **p** (str): The order of the norm to be used. Default is "fro" (Frobenius norm).
        - **log_term** (bool): Whether to include the log term in the loss computation. Default is False.
        - **alpha** (float): A scaling factor for the log term in the loss computation. Default is 1.0.
        - **form** (str): The form of the loss to be used. Default is None.
        - **noise_energy** (float): The energy of the noise to be used for masking. Default is None.

    References:
        - Yamamoto, R., Song, E., & Kim, J. M. (2020, May). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6199-6203). IEEE.
        - Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP: Differentiable digital signal processing. arXiv preprint arXiv:2001.04643.
    """

    def __init__(
        self,
        nfft: List[int] = [128, 256, 512, 1024, 2048, 4096],
        overlap: float = 0.75,
        sample_rate: int = 48000,
        energy_norm: bool = False,
        device="cpu",
        name: str = "MSS",
        apply_mask: bool = False,
        threshold: float = 5,
        p: str = "fro",
        log_term: bool = False,
        alpha: float = 1.0,
        form: str = None,
        noise_energy=None,
    ):
        super().__init__()
        self.nfft = nfft
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.energy_norm = energy_norm
        self.name = name
        self.device = device
        self.apply_mask = apply_mask
        self.threshold = threshold
        self.p = p
        self.log_term = log_term
        self.alpha = alpha
        self.form = form
        self.noise_energy = noise_energy

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        n_channels = y_pred.shape[-1]
        batch_size = y_pred.shape[0]

        if self.energy_norm:
            y_pred = y_pred / torch.norm(y_pred, p=2)
            y_true = y_true / torch.norm(y_true, p=2)

        # reshape it to (num_audio, len_audio) as indicated by nnAudio
        y_pred = torch.reshape(y_pred, (-1, y_pred.shape[1]))
        y_true = torch.reshape(y_true, (-1, y_true.shape[1]))

        loss = 0  # initialize match loss
        for i, nfft in enumerate(self.nfft):
            # initialize stft function with new nfft
            hop_length = int(nfft * (1 - self.overlap))
            lin_stft = features.stft.STFT(
                n_fft=nfft,
                hop_length=hop_length,
                window="hann",
                freq_scale="linear",
                sr=self.sample_rate,
                fmin=20,
                fmax=self.sample_rate // 2,
                output_format="Magnitude",
                verbose=False,
            )
            lin_stft = lin_stft.to(self.device).to(y_pred.dtype)

            h, w = tuple(lin_stft(y_pred).shape[-2:])
            Y_pred_lin = torch.reshape(lin_stft(y_pred),
                                       (batch_size, h, w, n_channels))
            Y_true_lin = torch.reshape(lin_stft(y_true),
                                       (batch_size, h, w, n_channels))
            Y_pred_log = torch.reshape(torch.log(lin_stft(y_pred)),
                                       (batch_size, h, w, n_channels))
            Y_true_log = torch.reshape(torch.log(lin_stft(y_true)),
                                       (batch_size, h, w, n_channels))

            mask = torch.ones_like(Y_true_lin)
            if self.apply_mask:
                if not self.noise_energy:
                    # compute the noise energy as the mean of the last 0.01s
                    self.noise_energy = torch.mean(
                        torch.pow(
                            Y_true_lin[:, :, -int(0.01 * self.sample_rate /
                                                  hop_length), :],
                            2,
                        ))
                SNR = 10 * torch.log10(
                    torch.max(Y_true_lin**2, self.noise_energy * 1.01) -
                    self.noise_energy) - 10 * torch.log10(self.noise_energy)
                mask[SNR < self.threshold] = 0
                N = torch.sum(mask)
            else:
                N = torch.numel(Y_true_lin)

            # update match loss
            if self.form is None:
                loss += torch.norm(
                    (Y_true_lin - Y_pred_lin) * mask, p=self.p) / N
                if self.log_term:
                    loss += (self.alpha * torch.norm(
                        (Y_true_log - Y_pred_log) * mask, p=self.p) / N)
            elif self.form == "yamamoto":
                loss += torch.norm(
                    (Y_true_lin - Y_pred_lin) * mask, p="fro") / torch.norm(
                        Y_true_lin, p="fro") + self.alpha * torch.norm(
                            (Y_true_log - Y_pred_log) * mask,
                            p=1) / torch.numel(Y_true_log)
            elif self.form == "magenta":
                loss += (torch.norm(
                    (Y_true_lin - Y_pred_lin) * mask, p=1) + self.alpha *
                         torch.sum(torch.abs(Y_true_log - Y_pred_log) * mask)
                         ) / torch.numel(Y_true_lin)

        return loss


class AveragePower(nn.Module):
    r"""
    Average Power Loss.

    This loss function computes the average power convergence between prediction and target.
    It calculates the normalized Frobenius norm of the difference between the windowed spectrograms of the predicted and true signals.

    Attributes:
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **name** (str): A name for the loss function. Default is "Average Power".
        - **stride** (int): The stride for the convolution operation. Default is (4,4).
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".

    References:
        - Dal Santo, Gloria, et al. "Similarity metrics for late reverberation." 2024 58th Asilomar Conference on Signals, Systems, and Computers. IEEE, 2024.
    """

    def __init__(
            self,
            energy_norm: bool = False,
            name: str = "Average Power",
            stride: tuple = (4, 4),
            device="cpu",
    ):
        super(AveragePower, self).__init__()
        self.name = name
        self.energy_norm = energy_norm
        self.stride = stride
        self.device = device

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        if self.energy_norm:
            y_pred = y_pred / torch.norm(y_pred, p=2)
            y_true = y_true / torch.norm(y_true, p=2)
        return self.average_power(y_pred, y_true)[0]

    def average_power(self, y_pred, y_true):
        # compute the magnitude spectrogram
        S1 = torch.abs(
            torch.stft(
                y_pred.squeeze(),
                n_fft=1024,
                hop_length=256,
                window=torch.hann_window(1024).to(self.device),
                return_complex=True,
            ))
        S2 = torch.abs(
            torch.stft(
                y_true.squeeze(),
                n_fft=1024,
                hop_length=256,
                window=torch.hann_window(1024).to(self.device),
                return_complex=True,
            ))

        # create 2d window
        win = self.window2d(
            torch.hann_window(64, dtype=S1.dtype, device=self.device))
        # convolve spectrograms with the window
        S1_win = F.conv2d(
            S1.unsqueeze(0).unsqueeze(0),
            win.unsqueeze(0).unsqueeze(0),
            stride=self.stride,
        ).squeeze()
        S2_win = F.conv2d(
            S2.unsqueeze(0).unsqueeze(0),
            win.unsqueeze(0).unsqueeze(0),
            stride=self.stride,
        ).squeeze()
        # compute the normalized difference between the two windowed spectrograms
        return (
            torch.norm(S2_win - S1_win, p="fro") /
            torch.norm(S1_win, p="fro") / torch.norm(S2_win, p="fro"),
            S1_win,
            S2_win,
        )

    def window2d(self, window):
        """create a 2D window from a given 1D window"""
        return window[:, None] * window[None, :]


## -------------------- ENERGY DECAY RELIEF LOSSES
class edr_loss(nn.Module):
    r"""
    Energy Decay Relief (EDR) Loss.

    This loss function computes the frequency-dependent loss on the mel-scale energy decay relief (EDR).

    Attributes:
        - **nfft** (int): The FFT size for the STFT computation. Default is 1024.
        - **overlap** (float): The overlap ratio for the STFT computation. Default is 0.5.
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
        - **name** (str): A name for the loss function. Default is "EDR".

    References:
        - Mezza, A. I., Giampiccolo, R., & Bernardini, A. (2024). Modeling the frequency-dependent sound energy decay of acoustic environments with differentiable feedback delay networks. In Proceedings of the 27th International Conference on Digital Audio Effects (DAFx24) (pp. 238-245).
    """

    def __init__(
        self,
        nfft: int = 1024,
        overlap: float = 0.5,
        sample_rate: int = 48000,
        energy_norm: bool = False,
        device: str = "cpu",
        name: str = "EDR",
    ):
        super().__init__()
        self.nfft = nfft
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.energy_norm = energy_norm
        self.win_length = int(0.020 * self.sample_rate)
        self.name = name
        self.device = device

    def discard_last_n_percent(self, x, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * x.shape[1]))
        out = x[:, 0:last_id, :]

        return out

    def schroeder_backward_int(self, x):
        # expected shape (batch_size, h, w, n_channels)
        # Backwards integral
        out = torch.flip(_finite_audio(x), dims=[-2])
        out = torch.cumsum(out**2, dim=-2)
        out = torch.flip(out, dims=[-2])
        out = torch.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)

        # Normalize to 1
        if self.energy_norm:
            norm_vals = torch.amax(out, dim=-2, keepdim=True).clamp_min(
                torch.finfo(out.dtype).eps)  # per frequency and channel
        else:
            norm_vals = torch.ones(out.shape, device=out.device)

        out = out / norm_vals

        return out, norm_vals

    def get_edr(self, x):
        # Remove filtering artefacts (last 5 permille)
        out = self.discard_last_n_percent(x, 0.5)
        # compute EDCs
        out = self.schroeder_backward_int(self.filterbank(out))[0]
        # get energy in dB
        out = 10 * torch.log10(out + 1e-32)

        return out

    def mel_stft(self, x):
        # compute the mel spectrogram
        mel_stft = features.mel.MelSpectrogram(
            n_fft=self.nfft,
            hop_length=int(self.win_length * (1 - self.overlap)),
            window="hann",
            win_length=self.win_length,
            sr=self.sample_rate,
            fmin=20,
            fmax=self.sample_rate // 2,
            n_mels=64,
            verbose=False,
        )
        # nnAudio creates its convolution kernels in float32. Match both
        # device and dtype to the signal: EDRLoss promotes its inputs to
        # float64 below, and Conv1d requires the kernels and input to agree.
        mel_stft = mel_stft.to(device=x.device, dtype=x.dtype)

        return mel_stft(x)

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        batch_size, _, n_channels = y_pred.shape
        # nnAudio accepts (audio, time). Move channels next to batch before
        # flattening, otherwise a reshape interleaves time samples from
        # different ambisonic components.
        y_pred = _finite_audio(y_pred).permute(0, 2,
                                               1).reshape(-1, y_pred.shape[1])
        y_true = _finite_audio(y_true).permute(0, 2,
                                               1).reshape(-1, y_true.shape[1])
        pred_mel = self.mel_stft(y_pred)
        true_mel = self.mel_stft(y_true)
        n_bands, n_frames = pred_mel.shape[-2:]
        Y_pred = pred_mel.reshape(batch_size, n_channels, n_bands,
                                  n_frames).permute(0, 2, 3, 1)
        Y_true = true_mel.reshape(batch_size, n_channels, n_bands,
                                  n_frames).permute(0, 2, 3, 1)

        eps = torch.finfo(Y_pred.dtype).eps
        floor_db = 10 * np.log10(eps)
        Y_pred_edr = torch.nan_to_num(
            10 *
            torch.log10(self.schroeder_backward_int(Y_pred)[0].clamp_min(eps)),
            nan=floor_db,
            posinf=0.0,
            neginf=floor_db)
        Y_true_edr = torch.nan_to_num(
            10 *
            torch.log10(self.schroeder_backward_int(Y_true)[0].clamp_min(eps)),
            nan=floor_db,
            posinf=0.0,
            neginf=floor_db)
        return torch.norm((Y_true_edr - Y_pred_edr).clamp(-120.0, 120.0), p=1) / \
            torch.norm(Y_true_edr, p=1).clamp_min(eps)


## -------------------- ENERGY DECAY CURVE LOSSES
class edc_loss(nn.Module):
    r"""
    Energy Decay Curve (EDC) Loss.

    This loss function computes the loss on energy decay curves (EDCs).
    It evaluates the similarity between the predicted and target EDCs, either in broadband or subband.

    Attributes:
        - **sample_rate** (int): The sampling rate of the audio signals. Default is 48000.
        - **nfft** (int): The FFT size. Default is 96000.
        - **is_broadband** (bool): Whether to compute the loss in broadband or subband. Default is False.
        - **n_fractions** (int): The number of fractional octave bands for subband analysis. Default is 1.
        - **energy_norm** (bool): Whether to normalize the energy of the input signals. Default is False.
        - **convergence** (bool): Whether to compute the normalized mean squared error. Default is False.
        - **clip** (bool): Whether to clip the EDCs at -60 dB. Default is False.
        - **name** (str): A name for the loss function. Default is "EDC".
        - **device** (str): The device to run the computations on (e.g., "cpu" or "cuda"). Default is "cpu".
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        is_broadband: bool = False,
        n_fractions: int = 1,
        energy_norm: bool = False,
        convergence: bool = False,
        clip: bool = False,
        name: str = "EDC",
        device: str = "cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.is_broadband = is_broadband
        self.n_fractions = n_fractions
        self.energy_norm = energy_norm
        self.convergence = convergence
        self.clip = clip
        self.name = name
        self.device = device
        self.discard_n = 0.5
        self.mse = nn.MSELoss(reduction="mean")

    def filterbank(self, x):
        impulse = torch.zeros(x.shape[1])
        impulse[0] = 1.0
        filter = torch.tensor(
            pf.dsp.filter.fractional_octave_bands(
                pf.Signal(impulse.numpy(), self.sample_rate),
                num_fractions=self.n_fractions,
                frequency_range=(63, 16000),
            ).freq.T).squeeze().to(device=x.device, dtype=x.dtype)
        y = torch.zeros(*x.shape,
                        filter.shape[1],
                        device=x.device,
                        dtype=x.dtype)

        for i_band in range(filter.shape[-1]):
            y[..., i_band] = torch.fft.irfft(
                torch.einsum(
                    "nfb,f->nfb",
                    torch.fft.rfft(x, dim=1, n=x.shape[1] * 2 - 1),
                    torch.nn.functional.pad(filter[:, i_band],
                                            (0, x.shape[1] - filter.shape[0])),
                ),
                dim=1,
                n=x.shape[1],
            )
        return y

    def discard_last_n_percent(self, x, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * x.shape[1]))
        out = x[:, 0:last_id, :]

        return out

    def schroeder_backward_int(self, x):

        # Backwards integral
        out = torch.flip(_finite_audio(x), dims=[1])
        out = torch.cumsum(out**2, dim=1)
        out = torch.flip(out, dims=[1])
        out = torch.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)

        # Normalize to 1
        if self.energy_norm:
            norm_vals = torch.amax(out, dim=1, keepdim=True).clamp_min(
                torch.finfo(out.dtype).eps)  # per channel (and band)
        else:
            norm_vals = torch.ones_like(out)

        out = out / norm_vals

        return out, norm_vals

    def get_edc(self, x):
        # Remove filtering artefacts (last 5 permille)
        out = self.discard_last_n_percent(x, self.discard_n)
        # compute EDCs
        if self.is_broadband:
            out = self.schroeder_backward_int(out)[0]
        else:
            out = self.schroeder_backward_int(self.filterbank(out))[0]
        # get energy in dB
        # A finite-length RIR can have exactly zero remaining energy at its
        # tail. Clamp before the logarithm so the subsequent MSE never sees
        # -inf (which otherwise turns the loss into NaN).
        eps = torch.finfo(out.dtype).eps
        floor_db = 10 * np.log10(eps)
        out = torch.nan_to_num(10 * torch.log10(out.clamp_min(eps)),
                               nan=floor_db,
                               posinf=0.0,
                               neginf=floor_db)

        return out

    def forward(self, y_pred, y_true):
        # assert that y_pred and y_true have the same shape = (n_batch, n_samples, n_channels)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.unsqueeze(0).unsqueeze(-1)
            y_true = y_true.unsqueeze(0).unsqueeze(-1)
        assert (y_pred.shape == y_true.shape) & (
            len(y_true.shape) == 3
        ), "y_pred and y_true must have the same shape (n_batch, n_samples, n_channels)"

        # compute the edcs
        y_pred_edc = self.get_edc(y_pred)
        y_true_edc = self.get_edc(y_true)

        if self.clip:
            try:
                clip_indx = torch.nonzero(
                    y_true_edc
                    < (torch.max(y_true_edc, dim=1, keepdim=True)[0] - 60),
                    as_tuple=True,
                )
                y_pred_edc[clip_indx] = -180
                y_true_edc[clip_indx] = -180
            except AttributeError:
                pass

        # compute normalized mean squared error on the EDCs
        # Bound very large EDC deviations so a transiently unstable recursive
        # model cannot create an infinite logarithmic-loss gradient.
        num = self.mse((y_pred_edc - y_true_edc).clamp(-120.0, 120.0),
                       torch.zeros_like(y_true_edc))
        den = torch.mean(torch.pow(y_true_edc, 2)).clamp_min(
            torch.finfo(y_true_edc.dtype).eps)
        if self.convergence:
            return num / den
        else:
            return num

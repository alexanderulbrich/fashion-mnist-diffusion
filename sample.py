"""
Generate Fashion-MNIST samples with annealed Langevin dynamics and watch the
denoising process live in a dashboard-style figure.

It uses a pretrained model by default (downloaded on first run).

Examples (run from the repository root):
    uv run sample.py                          # 4x4 grid, live window
    uv run sample.py --n 25 --seed 7          # 5x5 grid, reproducible
    uv run sample.py --save samples/run.gif   # also record an animation
    uv run sample.py --no-show --save samples/run.mp4   # headless, e.g. on a server

What the figure shows:
    * left:          all samples of the batch, updated live
    * top right:     the noise schedule (sigma per level) with the current level highlighted
    * middle right:  the scaled score norm sigma * ||s_theta(x, sigma)|| per Langevin step
    * bottom:        a timeline of sample #1 at the end of every noise level
"""
import argparse
import math
import os

import numpy as np
import torch

from model import ScoreNet

# Fashion-MNIST: greyscale, 28x28
IMG = 28
DIM = IMG * IMG
MODEL_URL = "https://drive.google.com/uc?id=1lFbjTqwgH2VoHrPVCKGDqstLEQJbebkh"
CLASSES_HINT = "T-shirt · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Boot"

# Colours of the dashboard (dark theme)
BG = "#0f1117"
PANEL = "#161a23"
FG = "#e6e6e6"
MUTED = "#8a8f98"
ACCENT = "#4cc9f0"
ACCENT2 = "#f72585"


# --------------------------------------------------------------------------- #
# Argument parsing / setup
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16, help="number of samples generated in parallel (default 16)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible samples")
    p.add_argument("--num-sigmas", type=int, default=10, help="number of noise levels (must match the model)")
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=100, help="Langevin steps per noise level (default 100)")
    p.add_argument("--epsilon", type=float, default=2e-5, help="base step size of Langevin dynamics")
    p.add_argument("--final-denoise", action="store_true",
                   help="apply one noise-free denoising step x + sigma_min^2 * score at the end (crisper images)")
    p.add_argument("--render-every", type=int, default=4,
                   help="redraw the figure every k Langevin steps (higher = faster sampling)")
    p.add_argument("--cmap", default="gray", help="matplotlib colormap, e.g. gray, bone, magma, viridis")
    p.add_argument("--save", default=None, help="record an animation to this path (.gif, or .mp4 if ffmpeg is installed)")
    p.add_argument("--fps", type=int, default=30, help="frames per second of the recorded animation")
    p.add_argument("--no-show", action="store_true", help="do not open a window (headless mode)")
    p.add_argument("--out", default=os.path.join("samples", "final_grid.png"),
                   help="where to store the final dashboard as PNG (default samples/final_grid.png)")
    p.add_argument("--device", default="auto", help="cpu, cuda, mps or auto")
    p.add_argument("--model-path", default=None, help="path to a trained state dict (default: pretrained model)")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(num_sigmas, device, model_path=None):
    model = ScoreNet(dim=DIM, num_sigmas=num_sigmas).to(device)

    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "score_model_mnist_pretrained.pt")
        if not os.path.exists(model_path):
            import gdown
            print("Model hasn't been downloaded yet. Starting download...")
            gdown.download(MODEL_URL, model_path, quiet=False)
            print("Done.")
        else:
            print("Model already exists. Skipping download.")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def grid_shape(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def make_mosaic(x, rows, cols, pad=2):
    """Arrange a batch (N, 784) into a single (H, W) image with thin gaps between the tiles."""
    imgs = x.detach().float().cpu().clamp(0, 1).view(-1, IMG, IMG).numpy()
    h = rows * IMG + (rows + 1) * pad
    w = cols * IMG + (cols + 1) * pad
    mosaic = np.full((h, w), np.nan, dtype=np.float32)  # NaN -> drawn in the background colour
    for k, img in enumerate(imgs):
        r, c = divmod(k, cols)
        y0 = pad + r * (IMG + pad)
        x0 = pad + c * (IMG + pad)
        mosaic[y0:y0 + IMG, x0:x0 + IMG] = img
    return mosaic


def style_axis(ax, title=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3a")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=FG, fontsize=10, loc="left", pad=8, fontweight="bold")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class Dashboard:
    def __init__(self, plt, sigmas, steps, n, cmap):
        self.plt = plt
        self.sigmas = sigmas.cpu().numpy()
        self.L = len(self.sigmas)
        self.T = steps
        self.total = self.L * self.T
        self.rows, self.cols = grid_shape(n)

        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(PANEL)  # colour of the gaps in the mosaic

        self.fig = plt.figure(figsize=(14, 8.2), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("Score-based diffusion · Fashion-MNIST")
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            3, 2 + self.L, height_ratios=[1, 1, 0.55], width_ratios=[2.2, 2.2] + [1] * self.L,
            left=0.035, right=0.985, top=0.86, bottom=0.04, hspace=0.45, wspace=0.25,
        )

        # --- header -------------------------------------------------------
        self.fig.text(0.035, 0.955, "Annealed Langevin Dynamics", color=FG, fontsize=18, fontweight="bold")
        self.fig.text(0.035, 0.925, f"Score network on Fashion-MNIST  ·  {CLASSES_HINT}", color=MUTED, fontsize=9)
        self.status = self.fig.text(0.985, 0.955, "", color=FG, fontsize=11, ha="right", family="monospace")
        # progress bar
        self.ax_prog = self.fig.add_axes([0.035, 0.895, 0.95, 0.008])
        self.ax_prog.set_xlim(0, 1)
        self.ax_prog.set_ylim(0, 1)
        self.ax_prog.axis("off")
        self.ax_prog.add_patch(plt.Rectangle((0, 0), 1, 1, color="#2a2f3a"))
        self.prog_bar = plt.Rectangle((0, 0), 0, 1, color=ACCENT)
        self.ax_prog.add_patch(self.prog_bar)

        # --- sample grid (left) -------------------------------------------
        self.ax_grid = self.fig.add_subplot(gs[0:2, 0:2])
        style_axis(self.ax_grid, f"Samples ({n})")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        empty = np.full((self.rows * IMG + (self.rows + 1) * 2, self.cols * IMG + (self.cols + 1) * 2), np.nan)
        self.im_grid = self.ax_grid.imshow(empty, cmap=cm, vmin=0, vmax=1, interpolation="nearest")

        # --- noise schedule (top right) -----------------------------------
        self.ax_sig = self.fig.add_subplot(gs[0, 2:])
        style_axis(self.ax_sig, "Noise schedule  σ")
        lv = np.arange(1, self.L + 1)
        self.sig_bars = self.ax_sig.bar(lv, self.sigmas, color="#2a2f3a", width=0.7)
        self.ax_sig.set_yscale("log")
        self.ax_sig.set_xticks(lv)
        self.ax_sig.set_xlabel("noise level")
        self.ax_sig.grid(axis="y", color="#2a2f3a", lw=0.5, alpha=0.6)
        self.ax_sig.set_axisbelow(True)

        # --- score norm (middle right) ------------------------------------
        self.ax_score = self.fig.add_subplot(gs[1, 2:])
        style_axis(self.ax_score, "Scaled score norm  σ·‖s(x, σ)‖  (batch mean)")
        self.ax_score.set_xlim(0, self.total)
        self.ax_score.set_xlabel("Langevin step")
        self.ax_score.grid(color="#2a2f3a", lw=0.5, alpha=0.6)
        for i in range(1, self.L):
            self.ax_score.axvline(i * self.T, color="#2a2f3a", lw=0.8, ls="--")
        self.score_line, = self.ax_score.plot([], [], color=ACCENT2, lw=1.4)
        self.score_hist = []

        # --- timeline (bottom) --------------------------------------------
        tl_axes_ax = self.fig.add_subplot(gs[2, 0:2])
        tl_axes_ax.axis("off")
        tl_axes_ax.text(0, 0.62, "Timeline of sample #1", color=FG, fontsize=10, fontweight="bold",
                        transform=tl_axes_ax.transAxes)
        tl_axes_ax.text(0, 0.38, "state after each noise level  →", color=MUTED, fontsize=9,
                        transform=tl_axes_ax.transAxes)
        self.tl_images = []
        for i in range(self.L):
            ax = self.fig.add_subplot(gs[2, 2 + i])
            style_axis(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"σ={self.sigmas[i]:.2g}", color=MUTED, fontsize=8, pad=3)
            im = ax.imshow(np.full((IMG, IMG), np.nan), cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            self.tl_images.append((ax, im))

    # ------------------------------------------------------------------ #
    def update(self, x, level, step, elapsed_steps):
        """Redraw the live parts of the figure."""
        self.im_grid.set_data(make_mosaic(x, self.rows, self.cols))

        frac = elapsed_steps / self.total
        self.prog_bar.set_width(frac)
        self.status.set_text(
            f"level {level + 1:>2}/{self.L}   step {step + 1:>3}/{self.T}   σ = {self.sigmas[level]:.4f}   {frac:6.1%}"
        )

        for i, bar in enumerate(self.sig_bars):
            bar.set_color(ACCENT if i == level else ("#3d6f80" if i < level else "#2a2f3a"))

        ys = np.asarray(self.score_hist)
        self.score_line.set_data(np.arange(1, len(ys) + 1), ys)
        if len(ys):
            lo, hi = np.nanmin(ys), np.nanmax(ys)
            margin = 0.05 * (hi - lo + 1e-8)
            self.ax_score.set_ylim(lo - margin, hi + margin)

    def finish_level(self, x, level):
        """Store the state of sample #1 in the timeline and highlight it."""
        ax, im = self.tl_images[level]
        im.set_data(x[0].detach().float().cpu().clamp(0, 1).view(IMG, IMG).numpy())
        for spine in ax.spines.values():
            spine.set_color(ACCENT)
            spine.set_linewidth(1.2)

    def done(self, x):
        self.im_grid.set_data(make_mosaic(x, self.rows, self.cols))
        self.prog_bar.set_width(1.0)
        self.prog_bar.set_color("#4ade80")
        self.status.set_text("done ✓")
        for bar in self.sig_bars:
            bar.set_color("#3d6f80")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # logarithmically decreasing noise levels
    sigmas = torch.exp(torch.linspace(np.log(args.sigma_max), np.log(args.sigma_min), args.num_sigmas)).to(device)
    model = load_model(args.num_sigmas, device, args.model_path)

    dash = Dashboard(plt, sigmas, args.steps, args.n, args.cmap)
    show = not args.no_show
    if show:
        plt.ion()
        plt.show(block=False)

    writer = None
    if args.save:
        from matplotlib import animation
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        if args.save.lower().endswith(".gif"):
            writer = animation.PillowWriter(fps=args.fps)
        else:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
        writer.setup(dash.fig, args.save, dpi=100)

    def render():
        if writer is not None:
            writer.grab_frame(facecolor=BG)
        if show:
            dash.fig.canvas.draw_idle()
            dash.fig.canvas.flush_events()
            plt.pause(0.001)

    def window_closed():
        return show and not plt.fignum_exists(dash.fig.number)

    # Start from uniform noise
    x = torch.rand((args.n, DIM), device=device)
    dash.update(x, 0, -1, 0)
    render()

    elapsed = 0
    aborted = False
    with torch.no_grad():
        for i in range(args.num_sigmas):
            sigma = sigmas[i]
            sigma_idx = torch.full((args.n,), i, dtype=torch.long, device=device)
            # effective step size, decreasing with the noise level
            alpha_i = args.epsilon * (sigma ** 2 / sigmas[-1] ** 2)
            for t in range(args.steps):
                score = model(x, sigma_idx)
                noise = torch.randn_like(x)
                # Langevin step: move along the score, plus injected noise
                x = x + alpha_i * score + torch.sqrt(2 * alpha_i) * noise
                elapsed += 1

                dash.score_hist.append((sigma * score.norm(dim=1)).mean().item())

                last_step = t == args.steps - 1
                if last_step:
                    dash.finish_level(x, i)
                if last_step or t % args.render_every == 0:
                    dash.update(x, i, t, elapsed)
                    render()
                    if window_closed():
                        aborted = True
                        break
            if aborted:
                print("Window closed - stopping.")
                break

        if not aborted and args.final_denoise:
            sigma_idx = torch.full((args.n,), args.num_sigmas - 1, dtype=torch.long, device=device)
            x = x + sigmas[-1] ** 2 * model(x, sigma_idx)

    if not aborted:
        dash.done(x)
        if writer is not None:
            # hold the final frame for ~2 seconds
            for _ in range(2 * args.fps):
                writer.grab_frame(facecolor=BG)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            dash.fig.savefig(args.out, dpi=150, facecolor=BG)
            print(f"Saved final figure to {args.out}")

    if writer is not None:
        writer.finish()
        print(f"Saved animation to {args.save}")

    if show and not aborted:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()

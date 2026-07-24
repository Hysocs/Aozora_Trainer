"""Unified illustration-detail detector and standalone preview GUI."""

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_illustration_detail_map(
    pil_image: Image.Image,
    sensitivity: float = 0.55,
    border_feather: float = 0.04,
    grain_rejection: float = 0.70,
) -> np.ndarray:
    """Return a fast [H,W] map of illustration lines and fine texture.

    Lines and texture are both local high-frequency changes, so one Laplacian
    response detects them together. A small neighbourhood average favors useful
    coherent detail over isolated pixel noise.
    """
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0.55)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # Random grain is high-frequency but generally disappears at a coarser
    # scale and has no stable local direction. Real contours tend to retain both.
    coarse_gray = cv2.GaussianBlur(gray, (0, 0), 1.35)
    coarse = np.abs(cv2.Laplacian(coarse_gray, cv2.CV_32F, ksize=3))
    coarse /= max(float(np.percentile(coarse, 99.0)), 1.0e-6)

    grad_x = cv2.Sobel(coarse_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(coarse_gray, cv2.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv2.blur(grad_x * grad_x, (7, 7))
    tensor_yy = cv2.blur(grad_y * grad_y, (7, 7))
    tensor_xy = cv2.blur(grad_x * grad_y, (7, 7))
    directional = np.sqrt(
        (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy ** 2
    ) / (tensor_xx + tensor_yy + 1.0e-6)
    line_support = np.clip(
        0.65 * np.clip(coarse, 0.0, 1.0) + 0.35 * directional,
        0.0,
        1.0,
    )
    grain_rejection = float(np.clip(grain_rejection, 0.0, 1.0))
    detail *= (1.0 - grain_rejection) + grain_rejection * line_support

    # Coherent clusters receive a small lift without restoring isolated noise.
    coherence = cv2.blur(detail, (5, 5))
    coherence /= max(float(np.percentile(coherence, 99.0)), 1.0e-6)
    detail *= 0.75 + 0.25 * np.clip(coherence, 0.0, 1.0)

    # One robust scaling pass. Higher sensitivity lowers the acceptance floor.
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    floor = float(np.percentile(detail, 88.0 - sensitivity * 48.0))
    ceiling = float(np.percentile(detail, 99.5))
    detail = np.clip((detail - floor) / max(ceiling - floor, 1.0e-6), 0.0, 1.0)

    # Canvas borders often appear as strong artificial lines. Fade the outer
    # fraction smoothly from zero to full weight instead of hard-cropping it.
    border_feather = float(np.clip(border_feather, 0.0, 0.25))
    feather_px = int(round(min(gray.shape) * border_feather))
    if feather_px > 0:
        y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
        distance = np.minimum.reduce(
            (
                np.broadcast_to(x, gray.shape),
                np.broadcast_to(y, gray.shape),
                np.broadcast_to(gray.shape[1] - 1 - x, gray.shape),
                np.broadcast_to(gray.shape[0] - 1 - y, gray.shape),
            )
        ).astype(np.float32)
        edge_weight = np.clip(distance / feather_px, 0.0, 1.0)
        edge_weight = edge_weight * edge_weight * (3.0 - 2.0 * edge_weight)
        detail *= edge_weight
    return np.clip(detail, 0.0, 1.0).astype(np.float32)


def generate_lineart_loss_map(
    pil_image: Image.Image,
    latent_h: int,
    latent_w: int,
    oversample: int = 4,
) -> torch.Tensor:
    """Convert the pixel detail map directly into a latent-grid loss mask.

    Peak response keeps thin lines detectable, while RMS and average pooling
    account for how much of each latent cell is actually occupied. This avoids
    turning every cell touched by one strong pixel into a maximum-weight cell.
    """
    detail = generate_illustration_detail_map(pil_image, sensitivity=0.55)
    return detail_map_to_latent_loss_map(detail, latent_h, latent_w)


def detail_map_to_latent_loss_map(detail, latent_h, latent_w):
    """Reduce a full-resolution detail map to the exact training loss grid."""
    source = torch.from_numpy(np.asarray(detail, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    output_size = (int(latent_h), int(latent_w))
    peak = F.adaptive_max_pool2d(source, output_size)
    energy = F.adaptive_avg_pool2d(source.square(), output_size).sqrt()
    density = F.adaptive_avg_pool2d(source, output_size)
    # A line crossing only a small part of an 8x8 latent footprint remains
    # visible through `peak`, but no longer makes the whole footprint red.
    latent_mask = peak * 0.25 + energy * 0.60 + density * 0.15
    return latent_mask.squeeze(0).to(dtype=torch.float16).contiguous()


def selected_trainer_vae_config():
    """Return the active GUI preset and its configured VAE settings."""
    state_path = PROJECT_ROOT / "configs" / "gui_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config_name = str(state.get("last_config", "")).strip()
    if not config_name:
        raise ValueError("The trainer GUI has no selected configuration.")
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.json"
    preset = json.loads(config_path.read_text(encoding="utf-8"))
    mode = str(preset.get("active_mode", "sdxl")).strip().lower()
    block = preset.get(mode, {})
    if mode != "anima":
        raise ValueError(
            "VAE latent preview currently supports Anima configs. "
            "The selected trainer config is SDXL."
        )
    vae_path = Path(str(block.get("anima_vae_path", "")).strip())
    if not vae_path.is_file():
        raise FileNotFoundError(f"Configured Anima VAE was not found: {vae_path}")
    return {
        "name": config_name,
        "mode": mode,
        "path": vae_path,
        "tiled": bool(block.get("anima_vae_caching_tiled", True)),
        "tile_size": tuple(block.get("anima_vae_caching_tile_size", [96, 96])),
        "tile_stride": tuple(block.get("anima_vae_caching_tile_stride", [72, 72])),
    }


def vae_detail_footprint(detail_map, vae, device, config):
    """Encode detail and blank images, returning normalized latent activation."""
    height, width = detail_map.shape
    vae_h, vae_w = max(8, height // 8 * 8), max(8, width // 8 * 8)
    resized = cv2.resize(detail_map, (vae_w, vae_h), interpolation=cv2.INTER_AREA)
    rgb = np.repeat(resized[None, None, :, :], 3, axis=1)
    images = np.concatenate((rgb, np.zeros_like(rgb)), axis=0)
    videos = torch.from_numpy(images).unsqueeze(2).mul_(2.0).sub_(1.0)
    dtype = next(vae.parameters()).dtype
    with torch.inference_mode():
        latents = vae.encode(
            videos.to(dtype=dtype),
            device=device,
            tiled=config["tiled"],
            tile_size=config["tile_size"],
            tile_stride=config["tile_stride"],
        ).float().cpu()
    # Channel RMS relative to a blank input visualizes where VAE features react.
    response = (latents[0] - latents[1]).square().mean(dim=0).sqrt().squeeze(0).numpy()
    ceiling = max(float(np.percentile(response, 99.0)), 1.0e-6)
    return np.clip(response / ceiling, 0.0, 1.0).astype(np.float32)


def _run_semantic_preview_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    class SemanticPreviewApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Aozora Semantic Detail Preview")
            self.root.geometry("1500x900")
            self.source_image = None
            self.source_path = None
            self.preview_refs = []
            self.pending_update = None
            self.vae = None
            self.vae_config = None
            self.vae_preview = None
            self.latent_loss_preview = None
            self.vae_loading = False

            toolbar = ttk.Frame(root, padding=8)
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Load image", command=self.load_image).pack(side="left")
            ttk.Button(toolbar, text="Save map", command=self.save_map).pack(side="left", padx=(8, 18))
            self.vae_button = ttk.Button(
                toolbar, text="Build configured VAE preview", command=self.build_vae_preview
            )
            self.vae_button.pack(side="left", padx=(0, 18))

            self.sensitivity_var = tk.DoubleVar(value=0.55)
            self.border_feather_var = tk.DoubleVar(value=0.04)
            self.grain_rejection_var = tk.DoubleVar(value=0.70)
            self.overlay_var = tk.DoubleVar(value=0.62)
            self._add_slider(toolbar, "Detail sensitivity", self.sensitivity_var)
            self._add_slider(
                toolbar, "Border feather", self.border_feather_var, maximum=0.15
            )
            self._add_slider(toolbar, "Grain rejection", self.grain_rejection_var)
            self._add_slider(toolbar, "Overlay", self.overlay_var)

            self.status = tk.StringVar(value="Load an image to inspect its line and detail importance map.")
            ttk.Label(root, textvariable=self.status, padding=(10, 0, 10, 8)).pack(fill="x")

            legend = ttk.Frame(root, padding=(12, 2, 12, 8))
            legend.pack(fill="x")
            ttk.Label(
                legend,
                text="Semantic mask → loss multiplier\n(training spatial strength = 1.0)",
                justify="left",
            ).pack(side="left", padx=(0, 12), anchor="n")
            scale_frame = ttk.Frame(legend)
            scale_frame.pack(side="left", anchor="n")
            legend_canvas = tk.Canvas(
                scale_frame,
                width=600,
                height=24,
                highlightthickness=1,
                highlightbackground="#777",
            )
            legend_canvas.grid(row=0, column=0, columnspan=4, sticky="ew")
            for x_pos in range(600):
                value = x_pos / 599.0
                bgr = cv2.applyColorMap(
                    np.array([[round(value * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO
                )[0, 0]
                color = f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
                legend_canvas.create_line(x_pos, 0, x_pos, 24, fill=color)
            legend_points = (
                ("Low\nmask 0.00\nloss 1.00×", "w"),
                ("Moderate\nmask 0.33\nloss 1.33×", "center"),
                ("High\nmask 0.67\nloss 1.67×", "center"),
                ("Maximum\nmask 1.00\nloss 2.00×", "e"),
            )
            for column, (text, anchor) in enumerate(legend_points):
                scale_frame.columnconfigure(column, weight=1, uniform="legend")
                ttk.Label(
                    scale_frame,
                    text=text,
                    justify="left" if anchor == "w" else "right" if anchor == "e" else "center",
                    anchor=anchor,
                    padding=(0, 3, 0, 0),
                ).grid(row=1, column=column, sticky="ew")
            ttk.Label(
                legend,
                text="VAE footprint colors show relative activation only;\nthey are not loss multipliers.",
                justify="left",
                padding=(18, 0, 0, 0),
            ).pack(side="left", anchor="n")

            preview_host = ttk.Frame(root)
            preview_host.pack(fill="both", expand=True)
            preview_host.rowconfigure(0, weight=1)
            preview_host.columnconfigure(0, weight=1)
            canvas = tk.Canvas(preview_host, highlightthickness=0)
            vertical = ttk.Scrollbar(preview_host, orient="vertical", command=canvas.yview)
            horizontal = ttk.Scrollbar(preview_host, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            vertical.grid(row=0, column=1, sticky="ns")
            horizontal.grid(row=1, column=0, sticky="ew")

            previews = ttk.Frame(canvas, padding=8)
            preview_window = canvas.create_window((0, 0), window=previews, anchor="nw")
            previews.bind(
                "<Configure>",
                lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.bind(
                "<MouseWheel>",
                lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"),
            )
            canvas.bind(
                "<Shift-MouseWheel>",
                lambda event: canvas.xview_scroll(-int(event.delta / 120), "units"),
            )
            # Retain the requested large workspace even when the window is smaller;
            # the scrollbars then expose the panels instead of shrinking them.
            canvas.itemconfigure(preview_window, width=1460)
            self.panels = []
            panel_titles = (
                "Original",
                "Semantic mask (Turbo colors)",
                "Overlay",
                "Actual training latent mask (absolute 1.0×–2.0×)",
                "VAE feature response (relative diagnostic — NOT loss)",
            )
            for index, title in enumerate(panel_titles):
                frame = ttk.LabelFrame(previews, text=title, padding=5)
                frame.grid(
                    row=index // 2,
                    column=index % 2,
                    sticky="nsew",
                    padx=6,
                    pady=6,
                )
                frame.configure(width=710, height=570)
                frame.grid_propagate(False)
                frame.rowconfigure(0, weight=1)
                frame.columnconfigure(0, weight=1)
                label = ttk.Label(frame, anchor="center")
                label.grid(row=0, column=0, sticky="nsew")
                label.bind("<Configure>", lambda _event: self.schedule_update())
                self.panels.append(label)
            previews.columnconfigure(0, weight=1)
            previews.columnconfigure(1, weight=1)

        def _add_slider(self, parent, title, variable, minimum=0.0, maximum=1.0):
            frame = ttk.Frame(parent)
            frame.pack(side="left", padx=8)
            ttk.Label(frame, text=title).pack(anchor="w")
            ttk.Scale(frame, variable=variable, from_=minimum, to=maximum, length=180,
                      command=lambda _value: self.schedule_update()).pack(side="left")
            value_label = ttk.Label(frame, width=5)
            value_label.pack(side="left", padx=(4, 0))

            def refresh(*_args):
                value_label.configure(text=f"{variable.get():.2f}")
            variable.trace_add("write", refresh)
            refresh()

        def load_image(self):
            path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with Image.open(path) as loaded:
                    self.source_image = loaded.convert("RGB").copy()
                self.source_path = Path(path)
                self.vae_preview = None
                self.latent_loss_preview = None
                self.update_previews()
            except Exception as exc:
                messagebox.showerror("Could not load image", str(exc))

        def make_map(self):
            return generate_illustration_detail_map(
                self.source_image,
                self.sensitivity_var.get(),
                self.border_feather_var.get(),
                self.grain_rejection_var.get(),
            )

        def schedule_update(self):
            if self.source_image is None:
                return
            if self.pending_update is not None:
                self.root.after_cancel(self.pending_update)
            self.pending_update = self.root.after(100, self.update_previews)

        @staticmethod
        def heatmap_from_map(detail_map):
            map_u8 = np.clip(detail_map * 255.0, 0, 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(map_u8, cv2.COLORMAP_TURBO)
            return Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))

        @staticmethod
        def fit_panel(image, label):
            fitted = image.copy()
            fitted.thumbnail(
                (max(200, label.winfo_width() - 10), max(200, label.winfo_height() - 10)),
                Image.Resampling.LANCZOS,
            )
            return fitted

        def update_previews(self):
            self.pending_update = None
            if self.source_image is None:
                return
            try:
                detail_map = self.make_map()
                heat = self.heatmap_from_map(detail_map)
                overlay = Image.blend(self.source_image, heat, float(self.overlay_var.get()))
                if self.latent_loss_preview is not None:
                    latent_h, latent_w = self.latent_loss_preview.shape
                    self.latent_loss_preview = (
                        detail_map_to_latent_loss_map(detail_map, latent_h, latent_w)
                        .squeeze(0)
                        .float()
                        .numpy()
                    )
                vae_view = (
                    self.heatmap_from_map(
                        cv2.resize(
                            self.vae_preview,
                            self.source_image.size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
                    if self.vae_preview is not None
                    else Image.new("RGB", self.source_image.size, (24, 24, 24))
                )
                latent_loss_view = (
                    self.heatmap_from_map(
                        cv2.resize(
                            self.latent_loss_preview,
                            self.source_image.size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
                    if self.latent_loss_preview is not None
                    else Image.new("RGB", self.source_image.size, (24, 24, 24))
                )
                self.preview_refs = []
                for label, preview in zip(
                    self.panels,
                    (self.source_image, heat, overlay, latent_loss_view, vae_view),
                ):
                    photo = ImageTk.PhotoImage(self.fit_panel(preview, label))
                    label.configure(image=photo)
                    self.preview_refs.append(photo)
                selected = float((detail_map >= 0.5).mean() * 100.0)
                self.status.set(
                    f"{self.source_path.name} — {self.source_image.width}x{self.source_image.height} — "
                    f"{selected:.1f}% above 50% importance"
                )
            except Exception as exc:
                self.status.set(f"Preview failed: {exc}")

        def build_vae_preview(self):
            if self.source_image is None:
                messagebox.showinfo("No image", "Load an image first.")
                return
            if self.vae_loading:
                return
            self.vae_loading = True
            self.vae_button.configure(state="disabled")
            self.status.set("Loading the selected trainer VAE and encoding the semantic outline...")
            detail_map = self.make_map().copy()

            def worker():
                try:
                    config = selected_trainer_vae_config()
                    if str(PROJECT_ROOT) not in sys.path:
                        sys.path.insert(0, str(PROJECT_ROOT))
                    from training_utils.anima.loader import load_anima_vae

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if self.vae is None or self.vae_config["path"] != config["path"]:
                        self.vae = load_anima_vae(
                            config["path"], dtype=torch.bfloat16, device=device
                        )
                        self.vae_config = config
                    footprint = vae_detail_footprint(detail_map, self.vae, device, config)
                    latent_loss = (
                        detail_map_to_latent_loss_map(
                            detail_map, footprint.shape[0], footprint.shape[1]
                        )
                        .squeeze(0)
                        .float()
                        .numpy()
                    )
                    self.root.after(
                        0, lambda: finish(footprint, latent_loss, config, None)
                    )
                except Exception as exc:
                    self.root.after(
                        0, lambda error=exc: finish(None, None, None, error)
                    )

            def finish(footprint, latent_loss, config, error):
                self.vae_loading = False
                self.vae_button.configure(state="normal")
                if error is not None:
                    self.status.set(f"VAE preview failed: {error}")
                    messagebox.showerror("VAE preview failed", str(error))
                    return
                self.vae_preview = footprint
                self.latent_loss_preview = latent_loss
                self.update_previews()
                self.status.set(
                    f"Latent comparison: {config['name']} | {config['path'].name} | "
                    f"grid {footprint.shape[1]}x{footprint.shape[0]} | "
                    f"actual mask mean {latent_loss.mean():.3f}, max {latent_loss.max():.3f}"
                )

            threading.Thread(target=worker, daemon=True).start()

        def save_map(self):
            if self.source_image is None:
                messagebox.showinfo("No image", "Load an image first.")
                return
            path = filedialog.asksaveasfilename(
                title="Save semantic map",
                defaultextension=".png",
                initialfile=f"{self.source_path.stem}_semantic.png",
                filetypes=[("PNG image", "*.png")],
            )
            if path:
                Image.fromarray(np.clip(self.make_map() * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
                self.status.set(f"Saved semantic map: {path}")

    root = tk.Tk()
    SemanticPreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    _run_semantic_preview_gui()

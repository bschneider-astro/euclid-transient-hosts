import argparse
import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ZScaleInterval

import aplpy

import warnings
from astropy.wcs import FITSFixedWarning
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Euclid cutouts produced by euclid_get_cutout.py"
    )
    parser.add_argument(
        "--cutout-folder",
        required=True,
        help="Base folder containing one subfolder per target with Euclid FITS cutouts."
    )
    parser.add_argument(
    "--mode",
    default="stacked",
    choices=["stacked", "mosaic"],
    help="Plotting mode: 'stacked' for VIS/NIR panels, 'mosaic' for all available ground+Euclid filters (default: stacked)."
    )
    parser.add_argument(
        "--plots-folder",
        default=None,
        help="Folder to store summary PNGs (default: <cutout-folder>/plots/)."
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.25,
        help="ZScale contrast for image display (default: 0.25)."
    )
    parser.add_argument(
        "--zoom-arcsec",
        type=float,
        default=50.0,
        help="Field width/height to show in arcsec (default: 50)."
    )
    parser.add_argument(
        "--poserr-fallback",
        type=float,
        default=2.5,
        help="Fallback positional error in arcsec if POSERR is missing (default: 2.5)."
    )
    parser.add_argument(
        "--idr-label",
        default="IDR1",
        help="Label for Euclid data release shown in panel titles (default: IDR1)."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)."
    )

    return parser.parse_args()


# =========================
# Helpers
# =========================
def build_file_dict(cutout_folder):
    """
    Scan subfolders under cutout_folder and collect Euclid FITS cutouts
    (those with '_euc_' in the filename).
    """
    result = {}
    subfolders = [
        f for f in os.listdir(cutout_folder)
        if os.path.isdir(os.path.join(cutout_folder, f))
    ]

    for sub in subfolders:
        sub_path = os.path.join(cutout_folder, sub)
        if not os.path.isdir(sub_path):
            continue

        euc_fits = [
            os.path.join(sub_path, f)
            for f in os.listdir(sub_path)
            if f.lower().endswith(".fits") and "_euc_" in f.lower()
        ]

        if euc_fits:
            result[sub] = sorted(euc_fits)

    return result


def plot_target_cutouts(grb, files, args, plots_dir, order):
    """
    Plot all cutouts for a single target (GRB or source),
    arranged in fixed panels: VIS, NIR_Y, NIR_J, NIR_H.
    Only the filters that are available are populated; the others are left blank.
    """
    logger.info(f"Processing {grb} with {len(files)} files...")

    contrast = args.contrast
    zoom_arcsec = args.zoom_arcsec
    fallback_err_arcsec = args.poserr_fallback
    idr_label = args.idr_label

    band_files = {}
    for fp in files:
        with fits.open(fp) as hdul:
            hdr = hdul[0].header

        inst = (hdr.get("INSTRUME") or "").upper()
        filt = (hdr.get("FILTER") or ("VIS" if inst == "VIS" else "")).upper()

        band = "VIS" if inst == "VIS" else filt  # e.g. VIS / NIR_Y / NIR_J / NIR_H
        if band and band not in band_files:
            band_files[band] = fp

    if not band_files:
        logger.warning(f"No valid FITS files found for {grb}, skipping.")
        return

    band_list = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
    n = len(band_list)

    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    available_bands = list(band_files.keys())
    available_bands.sort(key=lambda b: order.get(b, 99))
    meta_band = available_bands[0]
    meta_fp = band_files[meta_band]
    outdir = os.path.dirname(meta_fp)

    with fits.open(meta_fp) as hdul0:
        hdr0 = hdul0[0].header

    name = hdr0.get("OBJECT")
    redshift = hdr0.get("REDSHIFT")
    ra_targ = hdr0.get("RA_TARG")
    dec_targ = hdr0.get("DEC_TARG")

    if redshift is None or not np.isfinite(redshift):
        redshift = -99.99

    grb_coord = None
    if ra_targ is not None and dec_targ is not None:
        grb_coord = SkyCoord(ra_targ * u.deg, dec_targ * u.deg, frame="icrs")

    poserr_arcsec = hdr0.get("POSERR")
    if poserr_arcsec is not None and np.isfinite(poserr_arcsec):
        err_arcsec = float(poserr_arcsec)
    else:
        err_arcsec = float(fallback_err_arcsec)

    for i, band in enumerate(band_list):
        ax = axes[i]

        if band not in band_files:
            ax.set_facecolor("white")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.text(
                0.5, 0.5,
                f"No {band.replace('NIR_', 'NISP ')}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            continue
        else:
            ax.set_axis_off()

        fp = band_files[band]
        band_label = band.replace("NIR_", "NISP ")

        with fits.open(fp) as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)

        left = i * (1.0 / n)
        width = 1.0 / n
        bottom = 0.0
        height = 1.0

        f = aplpy.FITSFigure(
            fp,
            figure=fig,
            subplot=[left, bottom, width, height],
            axes=ax,
            north=True,
        )

        zscale = ZScaleInterval(contrast=contrast)
        vmin, vmax = zscale.get_limits(data)
        f.show_colorscale(cmap="gray_r", vmin=vmin, vmax=vmax)

        f.axis_labels.hide()
        f.tick_labels.hide()
        f.ticks.hide()

        if grb_coord is not None:
            grb_pixel = f.world2pixel(ra_targ, dec_targ)
            x, y = grb_pixel

            pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0
            radius_pixels = err_arcsec / pixel_scale
            circle = Circle(
                (x, y),
                radius_pixels,
                edgecolor="red",
                facecolor="none",
                lw=1.5,
            )
            f.ax.add_patch(circle)

            zoom_deg = zoom_arcsec / 3600.0
            f.recenter(ra_targ, dec_targ, width=zoom_deg, height=zoom_deg)

        f.add_label(
            0.5,
            0.92,
            f"Euclid {band_label} ({idr_label})",
            relative=True,
            size=18,
            horizontalalignment="center",
            color="black",
            bbox=dict(facecolor="xkcd:white", alpha=0.8),
        )

    if isinstance(name, str) and len(name) > 0:
        if name[0] == "G" and len(name) > 3:
            title = f"{name[:3]} {name[3:]} (R90 = {err_arcsec:.2f}″)"
        elif name[0] == "E" and len(name) > 2:
            title = f"{name[:2]} {name[2:]}"
        else:
            title = name
    else:
        title = grb

    fig.suptitle(title, fontsize=24, y=1.075)

    fileout_pdf = os.path.join(outdir, f"{name}_euclid_images.pdf")
    fileout_png = os.path.join(outdir, f"{name}_euclid_images.png")
    # fig.savefig(fileout_pdf, dpi=600, format="pdf", bbox_inches="tight")
    fig.savefig(fileout_png, dpi=300, format="png", bbox_inches="tight")

    summary_png = os.path.join(plots_dir, f"{name}_euclid.png")
    fig.savefig(summary_png, dpi=300, format="png", bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved plots for {name} -> {fileout_pdf}, {fileout_png}, {summary_png}")


def plot_mosaic_row(fig, row_bands, row_index, total_rows, band_files,
                    contrast, zoom_arcsec, grb_coord, ra_targ, dec_targ,
                    err_arcsec, idr_label):
    """
    Plot a single row of bands at the given row_index (0 = top).
    """
    ncols_row = len(row_bands)
    if ncols_row == 0:
        return

    row_height = 1.0 / total_rows
    bottom = 1.0 - (row_index + 1) * row_height

    for j, band in enumerate(row_bands):
        fp = band_files[band]

        with fits.open(fp) as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)

        width = 1.0 / ncols_row
        left = j * width
        height = row_height

        f = aplpy.FITSFigure(
            fp,
            figure=fig,
            subplot=[left, bottom, width, height],
            north=True,
        )

        zscale = ZScaleInterval(contrast=contrast)
        vmin, vmax = zscale.get_limits(data)
        f.show_colorscale(cmap="gray_r", vmin=vmin, vmax=vmax)

        f.axis_labels.hide()
        f.tick_labels.hide()
        f.ticks.hide()

        if grb_coord is not None:
            grb_pixel = f.world2pixel(ra_targ, dec_targ)
            x, y = grb_pixel

            pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0
            radius_pixels = err_arcsec / pixel_scale
            circle = Circle(
                (x, y),
                radius_pixels,
                edgecolor="red",
                facecolor="none",
                lw=1.5,
            )
            f.ax.add_patch(circle)

            zoom_deg = zoom_arcsec / 3600.0
            f.recenter(ra_targ, dec_targ, width=zoom_deg, height=zoom_deg)

        label = band
        if band.startswith("DECAM_"):
            label = "DECam " + band.split("_", 1)[1].lower()
        elif band.startswith("HSC_"):
            label = "HSC " + band.split("_", 1)[1].lower()
        elif band.startswith("MEGACAM_"):
            label = "MegaCam " + band.split("_", 1)[1].lower()
        elif band.startswith("PANSTARRS_"):
            label = "Pan-STARRS " + band.split("_", 1)[1].lower()
        elif band.startswith("NIR_"):
            label = "NISP " + band.split("_", 1)[1]
        elif band == "VIS":
            label = "VIS"

        f.add_label(
            0.5,
            0.92,
            f"{label} ({idr_label})",
            relative=True,
            size=12,
            horizontalalignment="center",
            color="black",
            bbox=dict(facecolor="xkcd:white", alpha=0.8),
        )

def plot_target_cutouts_mosaic(grb, files, args, plots_dir):
    """
    MOSAIC mode:
    - Bottom row: Euclid bands (VIS, NIR_Y, NIR_J, NIR_H) that are available.
    - Upper rows: ground-based bands (DECAM_*, HSC_*, MEGACAM_*, PANSTARRS_*, etc),
      automatically wrapped into multiple rows.
    """
    logger.info(f"Processing {grb} with {len(files)} files (mosaic mode)...")

    contrast = args.contrast
    zoom_arcsec = args.zoom_arcsec
    fallback_err_arcsec = args.poserr_fallback
    idr_label = args.idr_label

    band_files = {}
    for fp in files:
        with fits.open(fp) as hdul:
            hdr = hdul[0].header

        inst = (hdr.get("INSTRUME") or "").upper()
        filt = (hdr.get("FILTER") or "").upper()

        if filt:
            band = filt
        else:
            if inst == "VIS":
                band = "VIS"
            else:
                band = inst or "UNKNOWN"

        if band and band not in band_files:
            band_files[band] = fp

    if not band_files:
        logger.warning(f"No valid FITS files found for {grb}, skipping.")
        return

    euclid_list = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
    euclid_bands = [b for b in euclid_list if b in band_files]

    all_bands = list(band_files.keys())
    ground_bands = [b for b in all_bands if b not in euclid_list]

    ground_order = [
        "MEGACAM_U",
        "DECAM_G", "HSC_G",
        "DECAM_R", "HSC_R", "HSC_R2", "MEGACAM_R",
        "DECAM_I", "HSC_I", "HSC_I2", "PANSTARRS_I",
        "DECAM_Z", "HSC_Z",
    ]
    order_dict = {b: i for i, b in enumerate(ground_order)}

    ground_bands.sort(key=lambda b: order_dict.get(b, 999))

    ground_per_row = 4
    ground_rows = []
    if ground_bands:
        for i in range(0, len(ground_bands), ground_per_row):
            ground_rows.append(ground_bands[i:i + ground_per_row])

    has_euclid_row = len(euclid_bands) > 0
    total_rows = (1 if has_euclid_row else 0) + len(ground_rows)
    if total_rows == 0:
        logger.warning(f"No usable bands for {grb}, skipping.")
        return

    if euclid_bands:
        meta_band = euclid_bands[0]
    else:
        meta_band = ground_bands[0]

    meta_fp = band_files[meta_band]
    outdir = os.path.dirname(meta_fp)

    with fits.open(meta_fp) as hdul0:
        hdr0 = hdul0[0].header

    name = hdr0.get("OBJECT")
    redshift = hdr0.get("REDSHIFT")
    ra_targ = hdr0.get("RA_TARG")
    dec_targ = hdr0.get("DEC_TARG")

    if redshift is None or not np.isfinite(redshift):
        redshift = -99.99

    grb_coord = None
    if ra_targ is not None and dec_targ is not None:
        grb_coord = SkyCoord(ra_targ * u.deg, dec_targ * u.deg, frame="icrs")

    poserr_arcsec = hdr0.get("POSERR")
    if poserr_arcsec is not None and np.isfinite(poserr_arcsec):
        err_arcsec = float(poserr_arcsec)
    else:
        err_arcsec = float(fallback_err_arcsec)

    max_cols = 1
    if euclid_bands:
        max_cols = max(max_cols, len(euclid_bands))
    for row in ground_rows:
        max_cols = max(max_cols, len(row))

    fig = plt.figure(figsize=(max_cols * 4.0, total_rows * 4.0))

    current_row = 0
    for row_bands in ground_rows:
        plot_mosaic_row(
            fig=fig,
            row_bands=row_bands,
            row_index=current_row,
            total_rows=total_rows,
            band_files=band_files,
            contrast=contrast,
            zoom_arcsec=zoom_arcsec,
            grb_coord=grb_coord,
            ra_targ=ra_targ,
            dec_targ=dec_targ,
            err_arcsec=err_arcsec,
            idr_label=idr_label,
        )
        current_row += 1

    if euclid_bands:
        euclid_row_index = total_rows - 1
        plot_mosaic_row(
            fig=fig,
            row_bands=euclid_bands,
            row_index=euclid_row_index,
            total_rows=total_rows,
            band_files=band_files,
            contrast=contrast,
            zoom_arcsec=zoom_arcsec,
            grb_coord=grb_coord,
            ra_targ=ra_targ,
            dec_targ=dec_targ,
            err_arcsec=err_arcsec,
            idr_label=idr_label,
        )

    if isinstance(name, str) and len(name) > 0:
        if name[0] == "G" and len(name) > 3:
            title = f"{name[:3]} {name[3:]} (R90 = {err_arcsec:.2f}″)"
        elif name[0] == "E" and len(name) > 2:
            title = f"{name[:2]} {name[2:]}"
        else:
            title = name
    else:
        title = grb

    if total_rows == 1:
        y_title = 1.075
    else:
        y_title = 1.05

    fig.suptitle(title, fontsize=24, y=y_title)

    fileout_pdf = os.path.join(outdir, f"{name}_euclid_mosaic_images.pdf")
    fileout_png = os.path.join(outdir, f"{name}_euclid_mosaic_images.png")
    # fig.savefig(fileout_pdf, dpi=600, format="pdf", bbox_inches="tight")
    fig.savefig(fileout_png, dpi=300, format="png", bbox_inches="tight")

    summary_png = os.path.join(plots_dir, f"{name}_euclid_mosaic.png")
    fig.savefig(summary_png, dpi=300, format="png", bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved MOSAIC plots for {name} -> {fileout_pdf}, {fileout_png}, {summary_png}")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    cutout_folder = args.cutout_folder

    if not os.path.isdir(cutout_folder):
        logger.error(f"Cutout folder does not exist or is not a directory: {cutout_folder}")
        sys.exit(-1)

    if args.plots_folder is None:
        plots_dir = os.path.join(cutout_folder, "plots")
    else:
        plots_dir = args.plots_folder

    os.makedirs(plots_dir, exist_ok=True)

    file_dict = build_file_dict(cutout_folder)

    if not file_dict:
        logger.warning(f"No Euclid FITS cutouts found under {cutout_folder}")
        sys.exit(0)

    logger.info(f"Found {len(file_dict)} targets with cutouts to plot.")

    order = {"VIS": 0, "NIR_Y": 1, "NIR_J": 2, "NIR_H": 3}
    for grb, files in file_dict.items():
        if args.mode == "stacked":
            plot_target_cutouts(grb, files, args, plots_dir, order)
        elif args.mode == "mosaic":
            plot_target_cutouts_mosaic(grb, files, args, plots_dir)

    logger.info("All plots created successfully.")


if __name__ == "__main__":
    main()
import argparse
import os
import sys
import logging
import math
import csv
import shutil
from io import StringIO
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.esa.euclid.core import EuclidClass

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("astroquery")
logger.handlers.clear()
logger.propagate = False
logger = logging.getLogger(__name__)

log_buffer = StringIO()
buffer_handler = logging.StreamHandler(log_buffer)
buffer_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(buffer_handler)

# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Euclid cutouts from a Euclid query CSV."
    )

    parser.add_argument("--csv", required=True,
                        help="Euclid query CSV (output of query script)")

    parser.add_argument("--env", default="IDR",
                        choices=["IDR", "OTF", "REG", "PDR"],
                        help="Euclid environment (default: IDR)")

    parser.add_argument("--credentials-file", default="./cred.txt",
                        help="Euclid credentials file (default: ./cred.txt)")

    parser.add_argument("--outpath", default="../data/cutouts/",
                        help="Base output folder for cutouts")

    parser.add_argument("--cutout-radius", type=float, default=0.5,
                        help="Cutout radius in arcmin (default 0.5)")

    parser.add_argument("--name-column", default="name",
                        help="Column name for target name (default: name)")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    return parser.parse_args()

# =========================
# Euclid login
# =========================
def login_euclid(env, credentials_file):
    logger.info(f"Using Euclid environment: {env}")
    Euclid = EuclidClass(environment=env, show_server_messages=False)

    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        sys.exit(-1)

    logger.info("Logging in…")
    Euclid.login(credentials_file=credentials_file, verbose=False)

    with open(credentials_file, "r") as f:
        user = f.readline().strip()

    logger.info(f"Logged in as {user}")
    return Euclid

# =========================
# Utility functions
# =========================
def is_finite(x):
    try:
        if np.ma.isMaskedArray(x):
            if x is np.ma.masked:
                return False
            x = x.filled(np.nan)
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def set_float(hdr, key, value, comment, scale=1.0):
    try:
        v = float(value) * scale
    except Exception:
        return
    if math.isfinite(v):
        hdr[key] = (v, comment)


def determine_cutout_id(row, colnames):
    fname = str(row["file_name"])

    if "MOSAIC" in fname:
        if "tile_index" in colnames and row["tile_index"] is not None:
            return row["tile_index"]
        if "mosaic_product_oid" in colnames:
            return row["mosaic_product_oid"]
        return ""
    else:
        return row["observation_id"]


def write_metadata(hdr, row, colnames, cutout_id, coord, name):
    if "instrument_name" in colnames:
        instr = str(row["instrument_name"]).upper()
    else:
        instr = "UNKNOWN"

    if "filter_name" in colnames and row["filter_name"] is not None:
        filt = str(row["filter_name"]).upper()
    else:
        filt = "VIS" if instr == "VIS" else ""

    hdr["INSTRUME"] = (instr, "Instrument name")
    if filt:
        hdr["FILTER"] = (filt, "Filter name")

    if "right_ascension" in colnames and is_finite(row["right_ascension"]):
        set_float(hdr, "RA_SRC", row["right_ascension"], "Source RA [deg]")
    if "declination" in colnames and is_finite(row["declination"]):
        set_float(hdr, "DEC_SRC", row["declination"], "Source Dec [deg]")

    hdr["OBSID"] = (str(cutout_id), "Observation/tile ID")
    hdr["OBJECT"] = (str(name), "Target name")

    set_float(hdr, "RA_TARG", coord.ra.deg, "Cutout center RA [deg]")
    set_float(hdr, "DEC_TARG", coord.dec.deg, "Cutout center Dec [deg]")

    if "pos_error" in colnames and is_finite(row["pos_error"]):
        set_float(hdr, "POSERR", row["pos_error"], "Positional error [arcsec]")

    if "redshift" in colnames:
        val = row["redshift"]
        if val is not None and not np.ma.is_masked(val):
            hdr["REDSHIFT"] = (float(val), "Redshift")
        else:
            hdr["REDSHIFT"] = (-99.0, "Redshift missing")

# =========================
# Main cutout creation
# =========================
def create_cutouts(Euclid, tab, args):
    failed = []
    colnames = tab.colnames
    radius = args.cutout_radius * u.arcmin
    outbase = args.outpath
    namecol = args.name_column

    if namecol not in colnames:
        logger.error(f"Column '{namecol}' not in table: {colnames}")
        sys.exit(-1)

    for i, row in enumerate(tab):
        name = row[namecol]
        filepath = f"{row['file_path']}/{row['file_name']}"
        cutout_id = determine_cutout_id(row, colnames)

        coord = SkyCoord(
            ra=float(row["right_ascension"]) * u.deg,
            dec=float(row["declination"]) * u.deg,
            frame="icrs",
        )

        out_dir = os.path.join(outbase, str(name))
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{name}_{row['file_name']}"
        out_path = os.path.join(out_dir, out_name)

        try:
            saved = Euclid.get_cutout(
                file_path=filepath,
                instrument=row["instrument_name"],
                id=str(cutout_id),
                coordinate=coord,
                radius=radius,
                output_file=out_path,
            )
            logger.info(f"[{i+1}] Saved cutout -> {saved}")

            actual = saved[0] if isinstance(saved, (list, tuple)) else saved

            with fits.open(actual, mode="update") as hdul:
                hdr = hdul[0].header
                write_metadata(hdr, row, colnames, cutout_id, coord, name)
                hdul.flush()

        except Exception as e:
            logger.error(f"[{i+1}] FAILED {filepath}: {e}")
            failed.append([name, filepath, str(cutout_id), str(e)])
            continue

    if failed:
        fail_csv = os.path.join(outbase, "failed_downloads.csv")
        with open(fail_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "filepath", "cutout_id", "error"])
            writer.writerows(failed)
        logger.info(f"Saved {len(failed)} failed downloads to {fail_csv}")
    else:
        logger.info("All cutouts were successfully downloaded.")

# =========================
# Main
# =========================
def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    logger.info("Loading query table…")
    tab = Table.read(args.csv, format="csv")

    Euclid = login_euclid(args.env, args.credentials_file)

    logger.info(f"Creating cutouts for {len(tab)} rows…")
    create_cutouts(Euclid, tab, args)

    logger.info("Finished successfully.")

if __name__ == "__main__":
    main()

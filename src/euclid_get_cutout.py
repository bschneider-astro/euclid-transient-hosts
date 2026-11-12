import argparse
import os
import sys
import logging
import math
import csv
import shutil
import glob
import re
import time
from io import StringIO
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

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
                        help="Euclid query CSV (output of query script), or a folder "
                             "containing one; the latest *_euclid_query_*.csv is used")

    parser.add_argument("--frame-type", default=None,
                        help="When --csv is a folder, only consider query CSVs for this "
                             "frame type (e.g. stacked_frame, calibrated_frame, mosaic)")

    parser.add_argument("--env", default="IDR",
                        choices=["IDR", "OTF", "REG", "PDR"],
                        help="Euclid environment (default: IDR)")

    parser.add_argument("--credentials-file", default="./cred.txt",
                        help="Euclid credentials file (default: ./cred.txt)")

    parser.add_argument("--outpath", default=None,
                        help="Base output folder for cutouts (default: the --csv folder "
                             "if a folder is given, else ../data/cutouts/)")

    parser.add_argument("--cutout-radius", type=float, default=0.5,
                        help="Cutout radius in arcmin (default 0.5)")

    parser.add_argument("--name-column", default="name",
                        help="Column name for target name (default: name)")

    parser.add_argument("--retries", type=int, default=5,
                        help="Number of extra attempts for each cutout that fails during "
                             "the download (default 5 = 5 retries)")

    parser.add_argument("--retry-delay", type=float, default=3.0,
                        help="Seconds to wait before each retry (default 3)")

    parser.add_argument("--retry-failed", action="store_true",
                        help="Only retry the rows listed in failed_downloads.csv from a "
                             "previous run (looked up in --outpath unless --failed-csv is given)")

    parser.add_argument("--failed-csv", default=None,
                        help="Path to a failed_downloads.csv to retry (implies --retry-failed)")

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
QUERY_CSV_RE = re.compile(r"^(\d{8}_\d{6})_euclid_query_(.+)\.csv$")


def resolve_query_csv(path, frame_type=None):
    """Return path itself if it is a file, else the latest query CSV in the folder."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        logger.error(f"CSV path not found: {path}")
        sys.exit(-1)

    candidates = []
    for f in glob.glob(os.path.join(path, "*_euclid_query_*.csv")):
        m = QUERY_CSV_RE.match(os.path.basename(f))
        if m is None:
            continue
        if frame_type is not None and m.group(2) != frame_type:
            continue
        candidates.append((m.group(1), os.path.getmtime(f), f))

    if not candidates:
        suffix = f" with frame type '{frame_type}'" if frame_type else ""
        logger.error(f"No *_euclid_query_*.csv{suffix} found in {path}")
        sys.exit(-1)

    # Latest by timestamp prefix written by euclid_query.py, mtime as tie-breaker
    candidates.sort()
    if len(candidates) > 1:
        logger.info(f"Found {len(candidates)} query CSVs in {path}, using the latest")
    return candidates[-1][2]


def select_failed_rows(tab, failed_csv, namecol):
    """Keep only the rows of tab whose (name, file_path/file_name) is in failed_csv."""
    if not os.path.isfile(failed_csv):
        logger.error(f"No failed downloads file found: {failed_csv}")
        sys.exit(-1)

    failed = Table.read(failed_csv, format="csv")
    if len(failed) == 0:
        logger.info(f"{failed_csv} is empty, nothing to retry.")
        sys.exit(0)

    failed_keys = {(str(r["name"]), str(r["filepath"])) for r in failed}
    keep = [
        (str(row[namecol]), f"{row['file_path']}/{row['file_name']}") in failed_keys
        for row in tab
    ]
    sub = tab[np.array(keep, dtype=bool)]

    n_bursts = len({k[0] for k in failed_keys})
    logger.info(f"Retrying {len(sub)} failed cutouts for {n_bursts} targets from {failed_csv}")
    if len(sub) < len(failed_keys):
        logger.warning(f"{len(failed_keys) - len(sub)} entries in {failed_csv} were not "
                       f"found in the query table and will be skipped")
    return sub


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

class BadCutoutError(RuntimeError):
    """The archive returned a file, but its WCS does not contain the target."""


def check_cutout_wcs(path, coord):
    """
    Verify that the target position falls inside the downloaded cutout
    according to its WCS. The archive cutout service sometimes returns a
    small (e.g. 98x98 px) image whose WCS points at an unrelated part of the
    sky, typically when the target lies at or beyond the edge of the tile.
    Raises BadCutoutError if the cutout is unusable.
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        nx, ny = hdr.get("NAXIS1"), hdr.get("NAXIS2")
        try:
            x, y = WCS(hdr).celestial.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
        except Exception as exc:
            raise BadCutoutError(f"cannot evaluate cutout WCS: {exc}")

    inside = (nx and ny and np.isfinite(x) and np.isfinite(y)
              and -0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5)
    if not inside:
        raise BadCutoutError(
            f"target not inside returned cutout ({nx}x{ny} px, "
            f"CRVAL=({hdr.get('CRVAL1')}, {hdr.get('CRVAL2')}), "
            f"target pixel=({x}, {y})); likely off the tile edge"
        )


def quarantine(path):
    """Rename a bad cutout so downstream scripts (which look for *.fits) ignore it."""
    if path and os.path.isfile(path):
        bad = path + ".bad"
        os.replace(path, bad)
        logger.warning(f"Moved invalid cutout to {bad}")


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

    total = len(tab)
    for i, row in enumerate(tab):
        name = row[namecol]
        filepath = f"{row['file_path']}/{row['file_name']}"
        progress = f"[{i+1}/{total}]"
        logger.info(f"{progress} Downloading cutout for {name} "
                    f"({total - i - 1} remaining after this one)")
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

        n_attempts = 1 + max(0, args.retries)
        actual = None
        for attempt in range(1, n_attempts + 1):
            try:
                saved = Euclid.get_cutout(
                    file_path=filepath,
                    instrument=row["instrument_name"],
                    id=str(cutout_id),
                    coordinate=coord,
                    radius=radius,
                    output_file=out_path,
                )
                if not saved:
                    raise RuntimeError("archive returned no cutout (see server error above)")
                logger.info(f"{progress} Saved cutout -> {saved}")

                actual = saved[0] if isinstance(saved, (list, tuple)) else saved

                check_cutout_wcs(actual, coord)

                with fits.open(actual, mode="update") as hdul:
                    hdr = hdul[0].header
                    write_metadata(hdr, row, colnames, cutout_id, coord, name)
                    hdul.flush()
                break

            except BadCutoutError as e:
                # Deterministic server-side problem: retrying returns the same file
                logger.error(f"{progress} BAD CUTOUT {filepath}: {e}")
                quarantine(actual)
                failed.append([name, filepath, str(cutout_id), f"bad cutout: {e}"])
                break

            except Exception as e:
                if attempt < n_attempts:
                    logger.warning(f"{progress} Attempt {attempt}/{n_attempts} failed for "
                                   f"{filepath}: {e}; retrying in {args.retry_delay:g} s")
                    time.sleep(args.retry_delay)
                else:
                    logger.error(f"{progress} FAILED {filepath} after {n_attempts} "
                                 f"attempt(s): {e}")
                    failed.append([name, filepath, str(cutout_id), str(e)])

    if failed:
        fail_csv = os.path.join(outbase, "failed_downloads.csv")
        with open(fail_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "filepath", "cutout_id", "error"])
            writer.writerows(failed)
        logger.info(f"Saved {len(failed)} failed downloads to {fail_csv}")
    else:
        fail_csv = os.path.join(outbase, "failed_downloads.csv")
        if os.path.isfile(fail_csv):
            os.remove(fail_csv)
        logger.info("All cutouts were successfully downloaded.")

# =========================
# Main
# =========================
def main():
    args = parse_args()
    if args.retries < 0 or args.retry_delay < 0:
        logger.error("--retries and --retry-delay must be >= 0")
        sys.exit(-1)
    logger.setLevel(getattr(logging, args.log_level.upper()))

    csv_path = resolve_query_csv(args.csv, args.frame_type)
    if args.outpath is None:
        args.outpath = args.csv if os.path.isdir(args.csv) else "../data/cutouts/"
    logger.info(f"Cutouts will be written to {args.outpath}")

    logger.info(f"Loading query table {csv_path}…")
    tab = Table.read(csv_path, format="csv")

    if args.retry_failed or args.failed_csv:
        failed_csv = args.failed_csv or os.path.join(args.outpath, "failed_downloads.csv")
        if args.name_column not in tab.colnames:
            logger.error(f"Column '{args.name_column}' not in table: {tab.colnames}")
            sys.exit(-1)
        tab = select_failed_rows(tab, failed_csv, args.name_column)
        if len(tab) == 0:
            logger.info("No matching rows to retry.")
            return

    Euclid = login_euclid(args.env, args.credentials_file)

    logger.info(f"Creating cutouts for {len(tab)} rows…")
    create_cutouts(Euclid, tab, args)

    logger.info("Finished successfully.")

if __name__ == "__main__":
    main()

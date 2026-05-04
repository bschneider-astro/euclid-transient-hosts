import argparse
import os
import sys
import logging
import datetime
import shutil
from io import StringIO

import numpy as np
from astropy.table import Table, join, vstack
from astropy.io import ascii

from astroquery.esa.euclid.core import EuclidClass

# Local imports
import cutout_utils as ut


# =========================
# Logging setup
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
        description="Query Euclid archive for frames around targets "
                    "and save results to CSV."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--single",
        action="store_true",
        help="Run for a single target (use with --name, --ra, --dec, --err)."
    )
    group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file with columns: name, ra, dec, err (arcsec)."
    )
    parser.add_argument("--name", help="Target name (for --single).")
    parser.add_argument("--ra", type=float, help="RA in degrees (for --single).")
    parser.add_argument("--dec", type=float, help="Dec in degrees (for --single).")
    parser.add_argument("--err", type=float, help="Positional error in arcsec (for --single).")
    parser.add_argument(
        "--env",
        default="IDR",
        choices=["IDR", "OTF", "REG", "PDR"],
        help="Euclid environment (default: IDR)."
    )
    parser.add_argument(
        "--frame-type",
        default="mosaic_frame",
        choices=["mosaic_frame", "stacked_frame"],
        help="Frame type to query (default: mosaic_frame)."
    )
    parser.add_argument(
        "--instrument",
        action="append",
        default=None,
        help="Instrument(s) for stacked_frame mode (VIS or NISP). "
             "Can be given multiple times. Default: VIS and NISP."
    )
    parser.add_argument(
        "--filters",
        nargs="*",
        default=None,
        help="NISP filters (e.g. Y J H). Only used for stacked_frame/NISP."
    )
    parser.add_argument(
        "--outpath",
        default="../data/cutouts/",
        help="Output directory (default: ../data/cutouts/)."
    )
    parser.add_argument(
        "--ext",
        default="",
        help="Extension to append to output dir (default: '')."
    )
    parser.add_argument(
        "--credentials-file",
        default="./cred.txt",
        help="Path to Euclid credentials file (default: ./cred.txt)."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)."
    )

    args = parser.parse_args()

    if args.single:
        missing = []
        for key in ["name", "ra", "dec", "err"]:
            if getattr(args, key) is None:
                missing.append(key)
        if missing:
            parser.error(
                f"--single requires --name, --ra, --dec, --err (missing: {', '.join(missing)})"
            )

    return args


# =========================
# Euclid login
# =========================
def login_euclid(env, credentials_file):
    logger.info(f"Using Euclid environment: {env}")
    Euclid = EuclidClass(environment=env, show_server_messages=False)
    ut.set_env(env)

    if not os.path.exists(credentials_file):
        logger.error(f"Credentials file not found: {credentials_file}")
        sys.exit(-1)

    logger.info("Credentials file found. Logging in.")
    Euclid.login(credentials_file=credentials_file, verbose=False)

    with open(credentials_file, "r") as f:
        user_name = f.readline().strip()
    logger.info(f"Successfully logged in as {user_name}")

    return Euclid, user_name


# =========================
# Target table builders
# =========================
def build_targets_from_single(name, ra, dec, err):
    target = Table(
        [[name], [ra], [dec], [err]],
        names=("name", "right_ascension", "declination", "pos_error")
    )
    target["target_oid"] = np.arange(1, len(target) + 1)
    target["euclid_input_id"] = np.arange(1, len(target) + 1)
    return target


def build_targets_from_csv(csv_path):
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(-1)

    logger.info(f"Reading target list from {csv_path}")
    tab = Table.read(csv_path, format="csv")

    required_cols = {"name", "ra", "dec", "err"}
    if not required_cols.issubset(set(tab.colnames)):
        logger.error(
            f"CSV file must contain columns: {', '.join(sorted(required_cols))}. Found: {tab.colnames}"
        )
        sys.exit(-1)

    target = Table()
    target["name"] = tab["name"]
    target["right_ascension"] = tab["ra"]
    target["declination"] = tab["dec"]
    target["pos_error"] = tab["err"]
    if "id" in tab.colnames and len(np.unique(tab["id"])) == len(tab):
        target["target_oid"] = tab["id"]
    else:
        target["target_oid"] = np.arange(1, len(target) + 1)
    target["euclid_input_id"] = np.arange(1, len(target) + 1)

    logger.info(f"Loaded {len(target)} targets from CSV.")
    return target


# =========================
# Euclid user table upload
# =========================
def upload_sources_table(Euclid, user_name, target, tab_name="target"):
    tab_user = f"user_{user_name}.{tab_name}"
    sources_table = target[["euclid_input_id", "right_ascension", "declination"]]

    try:
        Euclid.delete_user_table(table_name=tab_user)
        logger.info(f"Existing user table '{tab_user}' found. Deleting it.")
    except Exception:
        logger.info(f"No existing user table '{tab_user}' to delete.")

    Euclid.upload_table(upload_resource=sources_table, table_name=tab_name)
    logger.info(f"Uploaded {len(sources_table)} sources to Euclid user table '{tab_user}'.")


# =========================
# Euclid query logic
# =========================
def normalize(joined, prefix, final_name, preferred_suffix=None):
    candidates = [c for c in joined.colnames if c.startswith(prefix)]

    preferred = f"{final_name}{preferred_suffix}" if preferred_suffix else None
    if preferred in candidates:
        keep = preferred
        joined.rename_column(keep, final_name)
        keep = final_name
    elif final_name in candidates:
        keep = final_name
    elif candidates:
        keep = candidates[0]
        joined.rename_column(keep, final_name)
        keep = final_name
    else:
        raise KeyError(f"No column starting with '{prefix}' found in joined table.")

    for c in candidates:
        if c != keep and c in joined.colnames:
            del joined[c]


def run_euclid_query(Euclid, target, env, frame_type, instrument_names,
                     filters, dtime_str, outpath, user_name):

    if frame_type == "stacked_frame":
        logger.info(f"Searching for sources in {frame_type}...")
        tab_list = []

        if instrument_names is None:
            instrument_names = ["VIS", "NISP"]

        for inst in instrument_names:
            logger.info(f"Querying instrument {inst}...")
            df = ut.getFiles(
                product_type=frame_type,
                sources="target",
                instrument_name=inst,
                nisp_filters=filters or [],
                segmentation_map_id=False,
                logger=logger,
            )
            if df is not None and len(df) > 0:
                tab_list.append(Table.from_pandas(df))
            else:
                logger.info(f"No results for instrument {inst}.")

        if not tab_list:
            logger.warning("No stacked-frame results found for any instrument.")
            return None, None

        tab = vstack(tab_list)

    elif frame_type == "mosaic_frame":
        logger.info("Searching for sources in mosaic frames...")
        df = ut.getFiles(
            product_type="mosaicAll",
            sources="target",
            logger=logger,
        )
        if df is None or len(df) == 0:
            logger.warning("No mosaic results found.")
            return None, None
        tab = Table.from_pandas(df)

    else:
        logger.error(f"Unsupported frame_type {frame_type}")
        return None, None

    if "euclid_input_id" not in tab.colnames:
        logger.error(
            "Archive result is missing 'euclid_input_id'. "
            "The uploaded source table must include this stable join key."
        )
        return None, None

    if "target_oid" in tab.colnames:
        tab.rename_column("target_oid", "archive_target_oid")

    joined = join(tab, target, keys="euclid_input_id", join_type="inner")

    normalize(joined, "right_ascension", "right_ascension", preferred_suffix="_2")
    normalize(joined, "declination", "declination", preferred_suffix="_2")

    base = ["target_oid", "name", "right_ascension", "declination", "pos_error"]
    extras = [c for c in joined.colnames if c not in base]
    joined = joined[base + extras]

    os.makedirs(outpath, exist_ok=True)

    out_full = os.path.join(outpath, f"{dtime_str}_euclid_query_{frame_type}.csv")
    joined.write(out_full, format="csv", overwrite=True)
    logger.info(f"Full query saved to {out_full}")

    unique_oids = np.unique(joined["target_oid"])
    mask = np.isin(target["target_oid"], unique_oids)
    target_found = target[mask]
    logger.info("Found %d targets.", len(target_found))
    target_found = target_found[["name", "right_ascension", "declination", "pos_error"]]

    out_summary = os.path.join(outpath, f"{dtime_str}_euclid_targets_{frame_type}.csv")
    target_found.write(out_summary, format="csv", overwrite=True)
    logger.info(f"Found targets saved to {out_summary}")

    return joined, target_found


# =========================
# Main
# =========================
def main():
    args = parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

    now = datetime.datetime.now()
    dtime_str = now.strftime("%Y%m%d_%H%M%S")
    logger.info(f"Script started at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.single:
        target = build_targets_from_single(args.name, args.ra, args.dec, args.err)
    else:
        target = build_targets_from_csv(args.csv)

    Euclid, user_name = login_euclid(args.env, args.credentials_file)

    if args.ext != '':
        outpath = os.path.join(args.outpath, dtime_str + "_" + args.ext)
    else:
        outpath = os.path.join(args.outpath, dtime_str)
    logger.info(f"Using output path: {outpath}")

    upload_sources_table(Euclid, user_name, target, tab_name="target")

    if args.instrument is None:
        instrument_names = ["VIS", "NISP"]
    else:
        instrument_names = args.instrument

    joined, target_found = run_euclid_query(
        Euclid=Euclid,
        target=target,
        env=args.env,
        frame_type=args.frame_type,
        instrument_names=instrument_names,
        filters=args.filters,
        dtime_str=dtime_str,
        outpath=outpath,
        user_name=user_name,
    )

    if joined is None:
        logger.warning("No results found. Nothing written.")
        sys.exit(0)
    else:
        with open(os.path.join(outpath, f"{dtime_str}.log"), "w") as f:
            f.write(log_buffer.getvalue())

        if not args.single:
            shutil.copy(args.csv, outpath)
            logger.info(f"Copied input csv {args.csv} to output path {outpath}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()

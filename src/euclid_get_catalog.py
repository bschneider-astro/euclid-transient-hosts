import argparse
import datetime
import logging
import math
import os
import re
import shutil
import sys
from io import StringIO

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u


# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

log_buffer = StringIO()
buffer_handler = logging.StreamHandler(log_buffer)
buffer_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(buffer_handler)


CATALOG_COLUMNS = [
    "object_id",
    "right_ascension",
    "declination",
    "segmentation_map_id",
    "vis_det",
    "flux_vis_psf",
    "fluxerr_vis_psf",
    "flux_detection_total",
    "fluxerr_detection_total",
    "flux_vis_sersic",
    "flux_y_sersic",
    "flux_j_sersic",
    "flux_h_sersic",
    "flag_vis",
    "flag_y",
    "flag_j",
    "flag_h",
    "variable_flag",
    "extended_flag",
    "extended_prob",
    "det_quality_flag",
    "segmentation_area",
    "semimajor_axis",
    "semimajor_axis_err",
    "position_angle",
    "ellipticity",
    "ellipticity_err",
    "kron_radius",
    "kron_radius_err",
]

MORPHOLOGY_COLUMNS = [
    "sersic_angle",
    "sersic_angle_err",
    "moment_20",
    "moment_20_err",
    "concentration",
    "concentration_err",
]


# =========================
# Arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Query Euclid MER catalog sources around targets from a cutout/query CSV "
            "and save catalog rows to CSV."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with target coordinates, e.g. output from euclid_query.py.",
    )
    parser.add_argument(
        "--env",
        default="IDR",
        choices=["IDR", "OTF", "REG", "PDR"],
        help="Euclid environment (default: IDR).",
    )
    parser.add_argument(
        "--credentials-file",
        default="./cred.txt",
        help="Euclid credentials file (default: ./cred.txt).",
    )
    parser.add_argument(
        "--outpath",
        default="../data/catalogs/",
        help="Base output folder for catalog query products.",
    )
    parser.add_argument(
        "--ext",
        default="",
        help="Extension appended to the timestamped output directory.",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Column with target names (default: name).",
    )
    parser.add_argument(
        "--ra-column",
        default="right_ascension",
        help="Column with target RA in degrees (default: right_ascension).",
    )
    parser.add_argument(
        "--dec-column",
        default="declination",
        help="Column with target Dec in degrees (default: declination).",
    )
    parser.add_argument(
        "--err-column",
        default="pos_error",
        help="Optional column with target positional error in arcsec (default: pos_error).",
    )
    parser.add_argument(
        "--target-id-column",
        default="target_oid",
        help="Optional stable target id column used for deduplication (default: target_oid).",
    )
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=30.0,
        help="Cone-search radius around each target in arcsec (default: 30).",
    )
    parser.add_argument(
        "--catalog-table",
        action="append",
        default=None,
        help=(
            "MER catalog table to query. Can be repeated. "
            "Default: mer_catalogue_deep."
        ),
    )
    parser.add_argument(
        "--morphology-table",
        default="none",
        help=(
            "MER morphology table to join. Use 'auto' to infer deep/wide, "
            "or 'none' to skip the morphology join (default: none)."
        ),
    )
    parser.add_argument(
        "--columns",
        default="default",
        choices=["default", "all"],
        help="Query the default compact column set or all catalog columns.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Rows to save in each per-target top catalog. Use 0 for all rows.",
    )
    parser.add_argument(
        "--sort-by",
        default="sep_arcsec",
        help="Column used to sort outputs when available (default: sep_arcsec).",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not write the combined catalog across all targets.",
    )
    parser.add_argument(
        "--copy-input",
        action="store_true",
        help="Copy the input CSV into the output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    return parser.parse_args()


# =========================
# Euclid login
# =========================
def login_euclid(env, credentials_file):
    try:
        from astroquery.esa.euclid.core import EuclidClass
    except ImportError as exc:
        raise ImportError(
            "astroquery is required to query the Euclid archive. "
            "Install it in the active environment before running catalog queries."
        ) from exc

    logger.info("Using Euclid environment: %s", env)
    euclid = EuclidClass(environment=env, show_server_messages=False)

    if not os.path.exists(credentials_file):
        logger.error("Credentials file not found: %s", credentials_file)
        sys.exit(-1)

    logger.info("Logging in.")
    euclid.login(credentials_file=credentials_file, verbose=False)

    with open(credentials_file, "r", encoding="utf-8") as f:
        user = f.readline().strip()
    logger.info("Logged in as %s", user)

    return euclid


# =========================
# Input handling
# =========================
def is_finite(value):
    try:
        if np.ma.isMaskedArray(value):
            if value is np.ma.masked:
                return False
            value = value.filled(np.nan)
        return math.isfinite(float(value))
    except Exception:
        return False


def safe_name(value):
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    return text or "target"


def validate_identifier(identifier, label):
    if identifier.lower() == "none":
        return identifier
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", identifier):
        raise ValueError(f"Invalid {label}: {identifier}")
    return identifier


def read_targets(args):
    if not os.path.exists(args.csv):
        logger.error("CSV file not found: %s", args.csv)
        sys.exit(-1)

    tab = Table.read(args.csv, format="csv")
    required = [args.name_column, args.ra_column, args.dec_column]
    missing = [col for col in required if col not in tab.colnames]
    if missing:
        logger.error("CSV is missing required columns: %s", ", ".join(missing))
        logger.error("Found columns: %s", tab.colnames)
        sys.exit(-1)

    keys = []
    targets = []
    for index, row in enumerate(tab):
        if not is_finite(row[args.ra_column]) or not is_finite(row[args.dec_column]):
            logger.warning("Skipping row %d with non-finite coordinates.", index)
            continue

        if args.target_id_column in tab.colnames:
            key = str(row[args.target_id_column])
        else:
            key = (
                str(row[args.name_column]),
                round(float(row[args.ra_column]), 8),
                round(float(row[args.dec_column]), 8),
            )

        if key in keys:
            continue

        keys.append(key)
        pos_error = np.nan
        if args.err_column in tab.colnames and is_finite(row[args.err_column]):
            pos_error = float(row[args.err_column])

        targets.append(
            {
                "target_oid": key if isinstance(key, str) else len(targets) + 1,
                "name": str(row[args.name_column]),
                "right_ascension": float(row[args.ra_column]),
                "declination": float(row[args.dec_column]),
                "pos_error": pos_error,
            }
        )

    target_tab = Table(rows=targets)
    logger.info("Loaded %d unique targets from %d CSV rows.", len(target_tab), len(tab))
    return target_tab


# =========================
# Query construction
# =========================
def infer_morphology_table(catalog_table):
    lower = catalog_table.lower()
    schema = ""
    table = lower
    if "." in lower:
        schema, table = lower.rsplit(".", 1)
        schema += "."

    if table.endswith("_deep") or "deep" in table:
        return f"{schema}mer_morphology_deep"
    if table.endswith("_wide") or "wide" in table:
        return f"{schema}mer_morphology_wide"
    return ""


def select_clause(use_all_columns, morphology_table):
    if use_all_columns:
        columns = ["cat.*"]
    else:
        columns = [f"cat.{col}" for col in CATALOG_COLUMNS]

    if morphology_table:
        columns.extend(f"morph.{col}" for col in MORPHOLOGY_COLUMNS)

    columns.append(
        "DISTANCE(cat.right_ascension, cat.declination, "
        "{ra}, {dec}) AS sep_deg"
    )
    return ",\n      ".join(columns)


def build_catalog_query(ra, dec, radius_deg, catalog_table, morphology_table, use_all_columns):
    select = select_clause(use_all_columns, morphology_table).format(ra=ra, dec=dec)
    join = ""
    if morphology_table:
        join = f"""
    JOIN
      {morphology_table} AS morph
    ON
      cat.object_id = morph.object_id"""

    return f"""
    SELECT
      {select}
    FROM
      {catalog_table} AS cat{join}
    WHERE
      CONTAINS(
        POINT('ICRS', cat.right_ascension, cat.declination),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
      ) = 1
    """


def run_host_search(
    euclid,
    target,
    catalog_table,
    morphology_table,
    radius_deg,
    use_all_columns,
):
    query = build_catalog_query(
        ra=target["right_ascension"],
        dec=target["declination"],
        radius_deg=radius_deg,
        catalog_table=catalog_table,
        morphology_table=morphology_table,
        use_all_columns=use_all_columns,
    )
    logger.debug("Query for %s:\n%s", target["name"], query)
    job = euclid.launch_job_async(query, verbose=False)
    result = job.get_results()
    logger.info(
        "%s / %s: %d catalog rows.",
        target["name"],
        catalog_table,
        len(result),
    )
    return result


# =========================
# Catalog post-processing
# =========================
def add_target_columns(tab, target, catalog_table):
    nrows = len(tab)
    tab["target_oid"] = np.full(nrows, str(target["target_oid"]), dtype="U128")
    tab["target_name"] = np.full(nrows, str(target["name"]), dtype="U128")
    tab["target_ra"] = np.full(nrows, float(target["right_ascension"]), dtype=float)
    tab["target_dec"] = np.full(nrows, float(target["declination"]), dtype=float)
    tab["target_pos_error"] = np.full(nrows, float(target["pos_error"]), dtype=float)
    tab["catalog_table"] = np.full(nrows, str(catalog_table), dtype="U128")
    return tab


def add_separation_arcsec(tab):
    if "sep_deg" in tab.colnames and "sep_arcsec" not in tab.colnames:
        tab["sep_arcsec"] = np.array(tab["sep_deg"], dtype=float) * 3600.0
    return tab


def sort_table(tab, sort_by):
    if len(tab) == 0 or sort_by not in tab.colnames:
        return tab

    values = np.array(tab[sort_by], dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, np.inf))
    return tab[order]


# =========================
# Output
# =========================
def write_target_catalog(tab, outdir, target_name, top_n):
    target_dir = os.path.join(outdir, safe_name(target_name))
    os.makedirs(target_dir, exist_ok=True)

    full_path = os.path.join(target_dir, f"{safe_name(target_name)}_catalog.csv")
    tab.write(full_path, format="csv", overwrite=True)

    if top_n > 0 and len(tab) > top_n:
        top_rows = tab[:top_n]
        top_path = os.path.join(
            target_dir,
            f"{safe_name(target_name)}_catalog_top{top_n}.csv",
        )
        top_rows.write(top_path, format="csv", overwrite=True)
    else:
        top_path = full_path

    logger.info("Saved target catalog: %s", full_path)
    return full_path, top_path


def target_missing_row(target, reason):
    return {
        "target_oid": str(target["target_oid"]),
        "name": str(target["name"]),
        "right_ascension": float(target["right_ascension"]),
        "declination": float(target["declination"]),
        "pos_error": float(target["pos_error"]),
        "reason": reason,
    }


def make_outdir(base, ext):
    dtime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{dtime_str}_{ext}" if ext else dtime_str
    outdir = os.path.join(base, folder)
    os.makedirs(outdir, exist_ok=True)
    return dtime_str, outdir


# =========================
# Main
# =========================
def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    catalog_tables = args.catalog_table or ["mer_catalogue_deep"]
    catalog_tables = [
        validate_identifier(table, "catalog table") for table in catalog_tables
    ]

    radius_deg = (args.radius_arcsec * u.arcsec).to_value(u.deg)
    use_all_columns = args.columns == "all"
    targets = read_targets(args)
    euclid = login_euclid(args.env, args.credentials_file)
    dtime_str, outdir = make_outdir(args.outpath, args.ext)
    logger.info("Using output path: %s", outdir)

    combined = []
    missing_targets = []
    for target in targets:
        target_tables = []
        query_failures = []
        for catalog_table in catalog_tables:
            if args.morphology_table.lower() == "none":
                morphology_table = ""
            elif args.morphology_table.lower() == "auto":
                morphology_table = infer_morphology_table(catalog_table)
            else:
                morphology_table = validate_identifier(
                    args.morphology_table, "morphology table"
                )

            try:
                result = run_host_search(
                    euclid=euclid,
                    target=target,
                    catalog_table=catalog_table,
                    morphology_table=morphology_table,
                    radius_deg=radius_deg,
                    use_all_columns=use_all_columns,
                )
            except Exception as exc:
                query_failures.append(f"{catalog_table}: {exc}")
                logger.error(
                    "%s / %s failed with morphology table '%s': %s",
                    target["name"],
                    catalog_table,
                    morphology_table or "none",
                    exc,
                )
                if morphology_table:
                    logger.info(
                        "Retrying %s / %s without morphology join.",
                        target["name"],
                        catalog_table,
                    )
                    try:
                        result = run_host_search(
                            euclid=euclid,
                            target=target,
                            catalog_table=catalog_table,
                            morphology_table="",
                            radius_deg=radius_deg,
                            use_all_columns=use_all_columns,
                        )
                    except Exception as retry_exc:
                        query_failures.append(f"{catalog_table} without morphology: {retry_exc}")
                        logger.error(
                            "%s / %s failed without morphology join: %s",
                            target["name"],
                            catalog_table,
                            retry_exc,
                        )
                        continue
                else:
                    continue

            result = add_target_columns(result, target, catalog_table)
            result = add_separation_arcsec(result)
            result = sort_table(result, args.sort_by)
            if len(result) == 0:
                continue
            target_tables.append(result)

        if not target_tables:
            logger.warning("No catalog rows for target %s.", target["name"])
            reason = "no_rows"
            if query_failures:
                reason = "query_failed; " + " | ".join(query_failures)
            missing_targets.append(target_missing_row(target, reason))
            continue

        target_catalog = vstack(target_tables, metadata_conflicts="silent")
        target_catalog = sort_table(target_catalog, args.sort_by)
        write_target_catalog(target_catalog, outdir, target["name"], args.top_n)
        combined.append(target_catalog)

    if combined and not args.no_combined:
        combined_tab = vstack(combined, metadata_conflicts="silent")
        combined_tab = sort_table(combined_tab, args.sort_by)
        combined_path = os.path.join(outdir, f"{dtime_str}_euclid_host_catalog.csv")
        combined_tab.write(combined_path, format="csv", overwrite=True)
        logger.info("Saved combined catalog: %s", combined_path)

    missing_path = os.path.join(outdir, f"{dtime_str}_euclid_targets_without_catalog.csv")
    if missing_targets:
        missing_tab = Table(rows=missing_targets)
    else:
        missing_tab = Table(
            names=("target_oid", "name", "right_ascension", "declination", "pos_error", "reason"),
            dtype=("U128", "U128", float, float, float, "U512"),
        )
    missing_tab.write(missing_path, format="csv", overwrite=True)
    logger.info("Saved %d targets without catalog rows to %s", len(missing_tab), missing_path)

    with open(os.path.join(outdir, f"{dtime_str}.log"), "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())

    if args.copy_input:
        shutil.copy(args.csv, outdir)
        logger.info("Copied input CSV to %s", outdir)

    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()

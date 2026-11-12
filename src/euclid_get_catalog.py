"""Query Euclid DR1 catalogues for transient host candidates.

For every target in the input CSV, a cone search is run on the deep MER
catalogues first and on the wide catalogues if nothing is found. The candidate
hosts are then completed with morphology, photo-z, classification and physical
parameters. Results are written per target to ``<outdir>/<name>/``:

- ``<name>_catalog.csv``: minimal cone-search result
- ``<name>_catalog_full.csv``: minimal result merged with the full host data
"""

import argparse
import datetime
import glob
import logging
import math
import os
import re
import sys
import time
from io import StringIO
from numbers import Real

import numpy as np
import pandas as pd
from astropy.table import Table


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


# =========================
# Catalogue definitions
# =========================
# DR1 tables exist in several schemas (e.g. catalogue.* and dr1_catalogue_mat_views.*)
SCHEMA = "catalogue."

HOST_COLUMNS = (
    "right_ascension",
    "declination",
    "object_id",
    "det_quality_flag",
    "flux_detection_total",
    "fluxerr_detection_total",
    "flux_h_sersic",
    "segmentation_map_id",
    "segmentation_area",
    "kron_radius",
    "ellipticity",
    "variable_flag",
    "point_like_prob",
    "spurious_flag",
    "semimajor_axis",
    "semimajor_axis_err",
    "position_angle",
    "flux_vis_2fwhm_aper",
    "flux_y_2fwhm_aper",
    "flux_j_2fwhm_aper",
    "flux_h_2fwhm_aper",
    "fluxerr_vis_2fwhm_aper",
    "fluxerr_y_2fwhm_aper",
    "fluxerr_j_2fwhm_aper",
    "fluxerr_h_2fwhm_aper",
    "vis_det",
    "flux_nir_stack_2fwhm_aper",
    "fluxerr_nir_stack_2fwhm_aper",
    "mumax_minus_mag",
    "flux_u_ext_decam_2fwhm_aper",
    "flux_u_ext_lsst_2fwhm_aper",
    "flux_u_ext_megacam_2fwhm_aper",
    "fluxerr_u_ext_decam_2fwhm_aper",
    "fluxerr_u_ext_lsst_2fwhm_aper",
    "fluxerr_u_ext_megacam_2fwhm_aper",
    "flux_g_ext_decam_2fwhm_aper",
    "flux_g_ext_hsc_2fwhm_aper",
    "flux_g_ext_jpcam_2fwhm_aper",
    "flux_g_ext_lsst_2fwhm_aper",
    "fluxerr_g_ext_decam_2fwhm_aper",
    "fluxerr_g_ext_hsc_2fwhm_aper",
    "fluxerr_g_ext_jpcam_2fwhm_aper",
    "fluxerr_g_ext_lsst_2fwhm_aper",
    "flux_r_ext_decam_2fwhm_aper",
    "flux_r_ext_lsst_2fwhm_aper",
    "flux_r_ext_megacam_2fwhm_aper",
    "fluxerr_r_ext_decam_2fwhm_aper",
    "fluxerr_r_ext_lsst_2fwhm_aper",
    "fluxerr_r_ext_megacam_2fwhm_aper",
    "flux_i_ext_decam_2fwhm_aper",
    "flux_i_ext_lsst_2fwhm_aper",
    "flux_i_ext_panstarrs_2fwhm_aper",
    "fluxerr_i_ext_decam_2fwhm_aper",
    "fluxerr_i_ext_lsst_2fwhm_aper",
    "fluxerr_i_ext_panstarrs_2fwhm_aper",
    "flux_z_ext_decam_2fwhm_aper",
    "flux_z_ext_hsc_2fwhm_aper",
    "flux_z_ext_panstarrs_2fwhm_aper",
    "flux_z_ext_lsst_2fwhm_aper",
    "fluxerr_z_ext_decam_2fwhm_aper",
    "fluxerr_z_ext_hsc_2fwhm_aper",
    "fluxerr_z_ext_panstarrs_2fwhm_aper",
    "fluxerr_z_ext_lsst_2fwhm_aper",
)

COMMON_FILTERS = (
    "cat.mumax_minus_mag > -2.6",
    "cat.mu_max < 35",
)

DEEP_CATALOGS = (
    {
        "table": SCHEMA + "mer_catalogue_deep_survey",
        "morphology_table": SCHEMA + "mer_morphology_deep_survey",
        "photo_z_table": SCHEMA + "phz_photo_z_deep_survey",
        "classification_table": SCHEMA + "phz_classification_deep_survey",
        "physical_table": SCHEMA + "phz_physical_parameters_deep_survey",
        "survey_tier": "deep",
        "specific_tier": "deep_survey",
        "filters": COMMON_FILTERS,
    },
    {
        "table": SCHEMA + "mer_catalogue_deep_mode",
        "morphology_table": SCHEMA + "mer_morphology_deep_mode",
        "photo_z_table": SCHEMA + "phz_photo_z_deep_mode",
        "classification_table": SCHEMA + "phz_classification_deep_mode",
        "physical_table": SCHEMA + "phz_physical_parameters_deep_mode",
        "survey_tier": "deep",
        "specific_tier": "deep_mode",
        "filters": COMMON_FILTERS,
    },
)

WIDE_CATALOGS = (
    {
        "table": SCHEMA + "mer_catalogue_wide_mode",
        "morphology_table": SCHEMA + "mer_morphology_wide_mode",
        "photo_z_table": (SCHEMA + "phz_photo_z_wide", SCHEMA + "phz_photo_z_wide_survey"),
        "classification_table": (SCHEMA + "phz_classification_wide", SCHEMA + "phz_classification_wide_survey"),
        "physical_table": (SCHEMA + "phz_physical_parameters_wide", SCHEMA + "phz_physical_parameters_wide_survey"),
        "survey_tier": "wide",
        "specific_tier": "wide_mode",
        "filters": ("cat.det_quality_flag < 130", *COMMON_FILTERS),
    },
    {
        "table": SCHEMA + "mer_catalogue_wide_survey",
        "morphology_table": SCHEMA + "mer_morphology_wide_survey",
        "photo_z_table": (SCHEMA + "phz_photo_z_wide", SCHEMA + "phz_photo_z_wide_survey"),
        "classification_table": (SCHEMA + "phz_classification_wide", SCHEMA + "phz_classification_wide_survey"),
        "physical_table": (SCHEMA + "phz_physical_parameters_wide", SCHEMA + "phz_physical_parameters_wide_survey"),
        "survey_tier": "wide",
        "specific_tier": "wide_survey",
        "filters": ("cat.det_quality_flag < 130", *COMMON_FILTERS),
    },
)

# Tables joined to the MER catalogue and their ADQL aliases
JOINED_TABLES = {
    "morphology_table": "morph",
    "photo_z_table": "phz",
    "classification_table": "classification",
    "physical_table": "phys",
}

FULL_HOST_COLUMNS = (
    "cat.object_id",
    "cat.segmentation_map_id",
    "cat.segmentation_area",
    "cat.vis_det",
    "cat.flux_vis_psf",
    "cat.fluxerr_vis_psf",
    "cat.flux_detection_total",
    "cat.fluxerr_detection_total",
    "cat.flux_vis_sersic",
    "cat.flux_y_sersic",
    "cat.flux_j_sersic",
    "cat.flux_h_sersic",
    "cat.flag_vis",
    "cat.flag_y",
    "cat.flag_j",
    "cat.flag_h",
    "cat.variable_flag",
    "cat.extended_flag",
    "cat.extended_prob",
    "cat.point_like_flag",
    "cat.point_like_prob",
    "cat.deblended_flag",
    "cat.blended_prob",
    "cat.binary_flag",
    "cat.det_quality_flag",
    "cat.fwhm",
    "cat.semimajor_axis_err",
    "cat.position_angle",
    "cat.ellipticity_err",
    "cat.kron_radius",
    "cat.kron_radius_err",
    "cat.gal_ebv",
    "cat.gal_ebv_err",
    "cat.flux_h_templfit",
    "cat.flux_vis_to_h_templfit",
    "morph.sersic_sersic_vis_radius",
    "morph.sersic_sersic_vis_radius_err",
    "morph.sersic_sersic_vis_index",
    "morph.sersic_sersic_vis_axis_ratio",
    "morph.sersic_sersic_nir_radius",
    "morph.sersic_sersic_nir_radius_err",
    "morph.sersic_sersic_nir_index",
    "morph.sersic_sersic_nir_axis_ratio",
    "morph.sersic_visnir_flags",
    "morph.sersic_visnir_reduced_chi2",
    "morph.sersic_angle_err",
    "morph.flux_radius",
    "morph.flux_radius_err",
    "morph.moment_20",
    "morph.moment_20_err",
    "morph.concentration",
    "morph.concentration_err",
    "morph.agn_yes",
    "morph.peculiar_yes",
    "morph.dwarf_yes",
    "morph.major_merger",
    "phz.phz_median",
    "phz.phz_classification",
    "phz.phz_pdf",
    "phz.phz_weight",
    "phz.phz_flags",
    "classification.phz_gal_prob",
    "classification.phz_star_prob",
    "classification.phz_qso_prob",
    "phys.phz_pp_median_sfr",
    "phys.phz_pp_median_sfhage",
    "phys.phz_pp_median_stellarmass",
    "phys.phz_pp_median_stellarmetallicity",
)


# =========================
# Arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Query Euclid DR1 catalogues (deep first, then wide) around targets "
            "from a CSV and save minimal and full host catalogues per target."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "CSV with target coordinates, or a folder containing one. For a folder, "
            "the latest *_euclid_targets_<frame>.csv written by euclid_query.py is "
            "used, falling back to the latest *_euclid_query_<frame>.csv."
        ),
    )
    parser.add_argument(
        "--frame-type",
        default=None,
        help=(
            "When --csv is a folder, only consider CSVs for this frame type "
            "(e.g. stacked_frame, calibrated_frame, mosaic)."
        ),
    )
    parser.add_argument(
        "--bestpos-csv",
        default=None,
        help=(
            "Optional CSV with improved positions. Rows matching a target name "
            "override its RA, Dec and positional error."
        ),
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
        "--outdir",
        default=None,
        help=(
            "Output folder; per-target catalogues go to <outdir>/<name>/ "
            "(default: folder of the input CSV)."
        ),
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Target name column in --csv (default: name).",
    )
    parser.add_argument(
        "--ra-column",
        default="right_ascension",
        help="Target RA column in degrees in --csv (default: right_ascension).",
    )
    parser.add_argument(
        "--dec-column",
        default="declination",
        help="Target Dec column in degrees in --csv (default: declination).",
    )
    parser.add_argument(
        "--err-column",
        default="pos_error",
        help="Optional positional error column in --csv (default: pos_error).",
    )
    parser.add_argument(
        "--bestpos-name-column",
        default="name",
        help="Target name column in --bestpos-csv (default: name).",
    )
    parser.add_argument(
        "--bestpos-ra-column",
        default="ra",
        help="RA column in --bestpos-csv (default: ra).",
    )
    parser.add_argument(
        "--bestpos-dec-column",
        default="dec",
        help="Dec column in --bestpos-csv (default: dec).",
    )
    parser.add_argument(
        "--bestpos-err-column",
        default="err",
        help="Positional error column in --bestpos-csv (default: err).",
    )
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=30.0,
        help="Cone-search radius around each target in arcsec (default: 30).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of attempts per archive query (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Base delay in seconds between query attempts (default: 5).",
    )
    parser.add_argument(
        "--force-catalog",
        action="store_true",
        help="Re-run the cone search even if <name>_catalog.csv exists.",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Re-query the full host data even if <name>_catalog_full.csv exists.",
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

    if not os.path.exists(credentials_file):
        logger.error("Credentials file not found: %s", credentials_file)
        sys.exit(-1)

    logger.debug("Logging in to Euclid environment %s.", env)
    euclid = EuclidClass(environment=env, show_server_messages=False)
    euclid.login(credentials_file=credentials_file, verbose=False)
    return euclid


# =========================
# Input handling
# =========================
def is_finite(value):
    try:
        if value is np.ma.masked:
            return False
        return math.isfinite(float(value))
    except Exception:
        return False


def read_best_positions(args):
    if not args.bestpos_csv:
        return {}
    if not os.path.exists(args.bestpos_csv):
        logger.error("Best-position CSV not found: %s", args.bestpos_csv)
        sys.exit(-1)

    tab = Table.read(args.bestpos_csv, format="csv")
    columns = [
        args.bestpos_name_column,
        args.bestpos_ra_column,
        args.bestpos_dec_column,
        args.bestpos_err_column,
    ]
    missing = [col for col in columns if col not in tab.colnames]
    if missing:
        logger.error("Best-position CSV is missing columns: %s", ", ".join(missing))
        sys.exit(-1)

    # first matching row wins, as in a linear search over the table
    best = {}
    for row in tab:
        name = str(row[args.bestpos_name_column])
        if name not in best:
            best[name] = (
                row[args.bestpos_ra_column],
                row[args.bestpos_dec_column],
                row[args.bestpos_err_column],
            )
    logger.info("Loaded %d best positions from %s", len(best), args.bestpos_csv)
    return best


# Files written by euclid_query.py: <YYYYmmdd_HHMMSS>_euclid_<kind>_<frame>.csv
INPUT_CSV_RE = re.compile(r"^(\d{8}_\d{6})_euclid_(targets|query)_(.+)\.csv$")
# Files this script writes into the same folder, never to be used as input
OWN_OUTPUT_FRAMES = {"without_catalog"}


def resolve_input_csv(path, frame_type=None):
    """Return path itself if it is a file, else the latest target CSV in the folder.

    Prefers the one-row-per-target *_euclid_targets_<frame>.csv and falls back
    to the per-frame *_euclid_query_<frame>.csv if no targets file exists.
    """
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        logger.error("CSV path not found: %s", path)
        sys.exit(-1)

    candidates = {"targets": [], "query": []}
    for f in glob.glob(os.path.join(path, "*_euclid_*_*.csv")):
        m = INPUT_CSV_RE.match(os.path.basename(f))
        if m is None:
            continue
        stamp, kind, frame = m.groups()
        if frame in OWN_OUTPUT_FRAMES:
            continue
        if frame_type is not None and frame != frame_type:
            continue
        candidates[kind].append((stamp, os.path.getmtime(f), f))

    for kind in ("targets", "query"):
        if candidates[kind]:
            # Latest by timestamp prefix, mtime as tie-breaker
            found = sorted(candidates[kind])
            if len(found) > 1:
                logger.info(
                    "Found %d *_euclid_%s_*.csv files in %s, using the latest",
                    len(found), kind, path,
                )
            return found[-1][2]

    suffix = f" with frame type '{frame_type}'" if frame_type else ""
    logger.error(
        "No *_euclid_targets_*.csv or *_euclid_query_*.csv%s found in %s",
        suffix, path,
    )
    sys.exit(-1)


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

    best_positions = read_best_positions(args)

    targets = []
    seen = set()
    for row in tab:
        name = str(row[args.name_column])
        if name in seen:
            continue
        seen.add(name)
        ra = row[args.ra_column]
        dec = row[args.dec_column]
        err = row[args.err_column] if args.err_column in tab.colnames else np.nan
        if name in best_positions:
            ra, dec, err = best_positions[name]
            logger.debug("Using best position for %s.", name)

        targets.append(
            {
                "name": name,
                "ra": float(ra) if is_finite(ra) else np.nan,
                "dec": float(dec) if is_finite(dec) else np.nan,
                "err": float(err) if is_finite(err) else np.nan,
            }
        )

    if len(targets) < len(tab):
        logger.info("Dropped %d duplicate rows for the same target name",
                    len(tab) - len(targets))
    logger.info("Loaded %d targets from %s", len(targets), args.csv)
    return targets


# =========================
# Catalogue discovery
# =========================
def list_archive_tables(euclid):
    tables = euclid.load_tables(only_names=True, verbose=False)
    return {table.get_qualified_name().lower() for table in tables}


def resolve_catalog(catalog, available):
    """Pick the existing table for each entry of a catalogue configuration.

    Entries may list several candidate names; the first one present in the
    archive is used. Returns None if the MER catalogue itself is missing, and
    sets joined tables that do not exist to None.
    """
    resolved = dict(catalog)
    for key in ("table", *JOINED_TABLES):
        candidates = catalog[key]
        if isinstance(candidates, str):
            candidates = (candidates,)
        resolved[key] = next((t for t in candidates if t.lower() in available), None)
        if resolved[key] is None:
            logger.debug("%s: no %s among %s", catalog["specific_tier"], key, candidates)

    if resolved["table"] is None:
        return None
    return resolved


def resolve_catalogs(euclid):
    """Return the deep and wide catalogues available in this environment."""
    available = list_archive_tables(euclid)
    deep = [c for c in (resolve_catalog(c, available) for c in DEEP_CATALOGS) if c]
    wide = [c for c in (resolve_catalog(c, available) for c in WIDE_CATALOGS) if c]

    for catalog in (*deep, *wide):
        missing = [key for key in JOINED_TABLES if catalog[key] is None]
        logger.info(
            "Using %s%s",
            catalog["table"],
            f" (missing: {', '.join(missing)})" if missing else "",
        )
    if not deep and not wide:
        raise RuntimeError("No MER catalogue tables found in this environment")
    return deep, wide


# =========================
# Query construction
# =========================
def validate_search_parameters(ra, dec, radius_deg):
    values = (ra, dec, radius_deg)
    if not all(isinstance(value, Real) and math.isfinite(value) for value in values):
        raise ValueError("RA, Dec, and radius must be finite numbers")
    if not 0.0 <= ra < 360.0:
        raise ValueError("RA must be in the range [0, 360) degrees")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("Dec must be in the range [-90, 90] degrees")
    if radius_deg <= 0.0:
        raise ValueError("Search radius must be greater than zero")


def join_clause(catalog, key):
    """LEFT JOIN on object_id, or an empty string if the table is missing."""
    if not catalog[key]:
        return ""
    alias = JOINED_TABLES[key]
    return (
        f"LEFT JOIN {catalog[key]} AS {alias} "
        f"ON cat.object_id = {alias}.object_id\n"
    )


def build_catalog_query(catalog, ra, dec, radius_deg, columns=HOST_COLUMNS):
    """Build an ADQL cone search for one catalogue."""
    select_columns = [f"cat.{column}" for column in columns]
    select_columns.extend(
        (
            f"'{catalog['survey_tier']}' AS survey_tier",
            f"'{catalog['specific_tier']}' AS specific_tier",
        )
    )
    where_clauses = [
        (
            "CONTAINS("
            "POINT('ICRS', cat.right_ascension, cat.declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})"
            ") = 1"
        ),
        *catalog["filters"],
    ]
    joins = "".join(
        join_clause(catalog, key) for key in ("morphology_table", "photo_z_table")
    )

    return (
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + f"\nFROM {catalog['table']} AS cat\n"
        + joins
        + "WHERE "
        + "\n  AND ".join(where_clauses)
    )


def build_tier_query(catalogs, ra, dec, radius_deg, columns=HOST_COLUMNS):
    """Combine the cone searches of one survey tier with UNION ALL."""
    if not catalogs:
        raise ValueError("At least one catalogue is required")
    return "\n\nUNION ALL\n\n".join(
        build_catalog_query(catalog, ra, dec, radius_deg, columns)
        for catalog in catalogs
    )


def build_full_host_query(object_ids, catalog):
    """Build an ADQL query joining catalogue, morphology and photo-z products."""
    object_ids = pd.Series(object_ids).dropna().drop_duplicates()
    try:
        numeric_ids = pd.to_numeric(object_ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("object_id values must be numeric") from exc

    if numeric_ids.empty:
        raise ValueError("At least one object_id is required")
    if (numeric_ids % 1 != 0).any():
        raise ValueError("object_id values must be integers")

    id_list = ", ".join(str(int(object_id)) for object_id in numeric_ids)
    # only select columns from tables that exist in this environment
    aliases = {"cat"} | {
        alias for key, alias in JOINED_TABLES.items() if catalog[key]
    }
    select_columns = (
        *(col for col in FULL_HOST_COLUMNS if col.split(".", 1)[0] in aliases),
        f"'{catalog['survey_tier']}' AS survey_tier",
        f"'{catalog['specific_tier']}' AS specific_tier",
    )
    joins = "".join(join_clause(catalog, key) for key in JOINED_TABLES)

    return (
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + f"\nFROM {catalog['table']} AS cat\n"
        + joins
        + f"WHERE cat.object_id IN ({id_list})"
    )


# =========================
# Query execution
# =========================
def execute_query(euclid, query, label, max_retries=3, retry_delay=5.0):
    """Run an ADQL query with linearly increasing retry delays."""
    if max_retries < 1:
        raise ValueError("max_retries must be a positive integer")
    if retry_delay < 0:
        raise ValueError("retry_delay cannot be negative")

    logger.debug("%s query:\n%s", label, query)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            job = euclid.launch_job_async(query, verbose=False)
            if job is None:
                raise RuntimeError("archive returned no job")
            return job.get_results().to_pandas()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break

            delay = attempt * retry_delay
            logger.warning(
                "%s query attempt %d/%d failed: %s; retrying in %.1f s",
                label,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"{label} query failed after {max_retries} attempts") from last_error


def run_host_search(
    euclid, deep_catalogs, wide_catalogs, ra, dec, radius_deg, max_retries=3, retry_delay=5.0
):
    """Cone search on the deep catalogues, falling back to wide if empty."""
    validate_search_parameters(ra, dec, radius_deg)

    result = pd.DataFrame()
    if deep_catalogs:
        deep_query = build_tier_query(deep_catalogs, ra, dec, radius_deg)
        result = execute_query(euclid, deep_query, "DEEP", max_retries, retry_delay)

    if result.empty and wide_catalogs:
        logger.info("No deep sources found; searching wide catalogues")
        wide_query = build_tier_query(wide_catalogs, ra, dec, radius_deg)
        result = execute_query(euclid, wide_query, "WIDE", max_retries, retry_delay)

    logger.info("Host search returned %d rows", len(result))
    return result


def get_full_host_data(euclid, catalogs, minimal_df, max_retries=3, retry_delay=5.0):
    """Fetch full catalogue, morphology and photo-z data for candidate hosts."""
    required_columns = {"object_id", "specific_tier"}
    missing_columns = required_columns.difference(minimal_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"minimal_df is missing required columns: {missing}")
    if minimal_df.empty:
        return pd.DataFrame()

    catalogs_by_tier = {catalog["specific_tier"]: catalog for catalog in catalogs}
    results = []
    for specific_tier, tier_df in minimal_df.groupby("specific_tier", sort=False):
        if specific_tier not in catalogs_by_tier:
            raise ValueError(f"Catalogue {specific_tier!r} not available in this environment")
        query = build_full_host_query(tier_df["object_id"], catalogs_by_tier[specific_tier])
        result = execute_query(euclid, query, specific_tier, max_retries, retry_delay)
        if not result.empty:
            results.append(result)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True, sort=False)


def merge_host_data(minimal_df, full_df, target):
    """Add the full host columns to the minimal result and tag the target."""
    if len(full_df) > 0:
        new_cols = [col for col in full_df.columns if col not in minimal_df.columns]
        df = minimal_df.merge(
            full_df[["object_id"] + new_cols], on="object_id", how="left"
        )
    else:
        df = minimal_df.copy()
    df["target_name"] = target["name"]
    df["target_ra"] = target["ra"]
    df["target_dec"] = target["dec"]
    df["target_pos_error"] = target["err"]
    return df


# =========================
# Output
# =========================
def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


# =========================
# Main
# =========================
def process_target(args, target, outdir, radius_deg, deep_catalogs, wide_catalogs):
    """Return the full host catalogue of one target, or None if there is none."""
    # same folder layout as euclid_get_cutout.py, so catalogues sit next to the cutouts
    name = str(target["name"])
    target_dir = os.path.join(outdir, name)
    catalog_path = os.path.join(target_dir, f"{name}_catalog.csv")
    full_path = os.path.join(target_dir, f"{name}_catalog_full.csv")

    if os.path.exists(full_path) and not (args.force_catalog or args.force_full):
        logger.info("Full catalog for %s already exists. Reading it.", target["name"])
        return pd.read_csv(full_path)

    # sessions expire on long runs, so log in again for every target
    euclid = login_euclid(args.env, args.credentials_file)

    if os.path.exists(catalog_path) and not args.force_catalog:
        logger.info("Catalog for %s already exists. Reading it.", target["name"])
        minimal_df = pd.read_csv(catalog_path)
    else:
        minimal_df = run_host_search(
            euclid,
            deep_catalogs,
            wide_catalogs,
            ra=target["ra"],
            dec=target["dec"],
            radius_deg=radius_deg,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        if len(minimal_df) > 0:
            save_csv(minimal_df, catalog_path)

    if len(minimal_df) == 0:
        return None

    full_df = get_full_host_data(
        euclid,
        deep_catalogs + wide_catalogs,
        minimal_df,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    df = merge_host_data(minimal_df, full_df, target)
    save_csv(df, full_path)
    return df


def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    input_path = args.csv
    args.csv = resolve_input_csv(input_path, args.frame_type)
    if args.outdir:
        outdir = args.outdir
    elif os.path.isdir(input_path):
        outdir = input_path
    else:
        outdir = os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)
    logger.info("Using output path: %s", outdir)

    radius_deg = args.radius_arcsec / 3600.0
    targets = read_targets(args)
    deep_catalogs, wide_catalogs = resolve_catalogs(
        login_euclid(args.env, args.credentials_file)
    )

    combined = []
    missing_targets = []
    for i, target in enumerate(targets):
        logger.info("Processing %s... %d/%d", target["name"], i + 1, len(targets))
        try:
            df = process_target(
                args, target, outdir, radius_deg, deep_catalogs, wide_catalogs
            )
        except Exception as exc:
            logger.error("%s failed: %s", target["name"], exc)
            missing_targets.append({**target, "reason": f"query_failed; {exc}"})
            continue

        if df is None:
            logger.warning("No catalog found for %s.", target["name"])
            missing_targets.append({**target, "reason": "no_rows"})
            continue
        combined.append(df)

    dtime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if combined:
        combined_df = pd.concat(combined, ignore_index=True, sort=False)
        save_csv(combined_df, os.path.join(outdir, f"{dtime_str}_euclid_host_catalog.csv"))

    missing_df = pd.DataFrame(
        missing_targets, columns=["name", "ra", "dec", "err", "reason"]
    )
    save_csv(missing_df, os.path.join(outdir, f"{dtime_str}_euclid_targets_without_catalog.csv"))

    logger.info(
        "Finished processing %d targets. Found catalogs for %d, no catalogs for %d.",
        len(targets),
        len(combined),
        len(missing_targets),
    )
    if missing_targets:
        logger.info("Targets without catalogs: %s", [t["name"] for t in missing_targets])

    with open(os.path.join(outdir, f"{dtime_str}_get_catalog.log"), "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())


if __name__ == "__main__":
    main()

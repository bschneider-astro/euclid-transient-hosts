# euclid-transient-hosts
Python scripts for the Euclid Supernovae and Transients Working Group

# Installation

```
conda create -n euclid python=3.12
conda activate euclid
conda install astropy numpy scipy matplotlib astroquery pandas scikit-image tqdm
conda install conda-forge::aplpy
```

# Usage

## Queyring database 

### CSV file with multiple sources
```
python euclid_query.py \
  --csv ../test/cat.csv \
  --frame-type  stacked_frame \
  --env IDR \
  --outpath ../test/cutouts/ 
```

### Single object
```
python euclid_query.py \
  --single \
  --name GRB160821B \
  --ra 279.97621 \
  --dec 62.39144 \
  --err 2.2 \
  --frame-type mosaic_frame \
  --env IDR \
  --outpath ../test/cutouts/
```

## Getting cutouts
```
python euclid_get_cutout.py \
  --csv ../test/cat.csv/euclid_query_stacked_frame_YYYYMMDD_HHMMSS.csv \
  --env IDR \
  --outpath ../test/cutouts/
```

## Getting catalogs
```
python euclid_get_catalog.py \
  --csv ../test/cutouts/20260430_211719_euclid_query_stacked_frame.csv \
  --env IDR \
  --radius-arcsec 30 \
  --catalog-table mer_catalogue_deep \
  --outpath ../test/catalogs/
```

## Plotting cutout

### For stacked frames
```
python euclid_plot_cutout.py \
  --cutout-folder ../test/cutouts/ \
  --mode stacked
```

### For mosaic frames (MER)
```
python euclid_plot_cutout.py \
  --cutout-folder ../test/cutouts/ \
  --mode mosaic
```

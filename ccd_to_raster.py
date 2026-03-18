"""
# TAREFA 3

PURPOSE:
This script processes parquet files containing change detection results from satellite imagery analysis.
It filters and aggregates pixel-level change detection data, validates breaks using NDVI loss calculations,
converts the data to a multi-band raster format, and creates visualization files for use in GIS software like QGIS.

MAIN FUNCTIONALITY:
- Reads multiple parquet files containing change detection segments with break points
- Filters data by date range (optional) - only breaks within the date range are considered
- Processes each pixel to identify the most recent valid vegetation loss break:
    * Iterates through segments in reverse chronological order (newest to oldest)
    * Validates breaks using NDVI loss calculation between consecutive segments
    * Classifies each pixel as: valid break (is_break=1), no break (is_break=0), or uncertain break (is_break=-1)
- Converts filtered point data to a 4-band georeferenced raster (GeoTIFF)
- Creates QGIS style files for visualization of Band 1 (break dates)
- Optionally saves filtered points as a vector file

INPUTS:
- input_directory: Directory containing parquet files with columns:
  * x_coord, y_coord: UTM coordinates (EPSG:32629 assumed)
  * tBreak: Break date as milliseconds since Unix epoch (UTC)
  * tEnd: Segment end date as milliseconds since Unix epoch (UTC)
  * nirEnd, redEnd: NIR and Red band values at segment end (for NDVI calculation)
  * Other columns used by ndvi_loss_calculation function
- date_ranges: List of tuples with (start_date, end_date) for filtering (format: 'YYYY-MM-DD')
  * A separate raster is created for each date range
- boundary_shapefile: Optional shapefile path for spatial filtering
  * Pixels outside boundary are marked as no-break (is_break=0) in the output raster
- start_bands, end_bands: Lists of additional band names in the parquet files to include in the output raster (values at segment start and end)

OUTPUTS:
- Multi-band GeoTIFF raster file (.tif):
  * Band 1: last_tEnd (segment end dates in YYYYMMDD format)
    * Valid/uncertain breaks: YYYYMMDD integer value
    * Pixels with no breaks: 0
    * Pixels with no data: -9999 (NoData)
  * Band 2: last_tBreak (break dates in YYYYMMDD format)
    * Valid/uncertain breaks: YYYYMMDD integer value
    * Pixels with no breaks: 0
    * Pixels with no data: -9999 (NoData)
  * Band 3: is_break (break classification)
    * 1: valid break (confirmed vegetation loss via NDVI)
    * 0: no break detected
    * -1: uncertain break (tBreak != tEnd, needs validation)
    * -99: NoData
  * Band 4: ndvi_last_segment (NDVI value of last segment, scaled by 10000)
    * Integer value for pixels with breaks (divide by 10000 to get original NDVI)
    * -9999 for pixels without breaks or NoData
  * start_bands, end_bands : (MC; feb 2026) - additional bands (greenStart, greenEnd, ...) can be added if needed for further analysis/visualization
    * Values for these bands are taken from the segment start and end values in the parquet files, and are set to NaN for pixels without breaks
  * Resolution: 10m x 10m pixels
  * Coordinate system: UTM (EPSG:32629) or optionally reprojected
- QGIS style file (.qml): Color-coded visualization of Band 1 by year with gradient by day-of-year
- Optional vector file (.gpkg): Point locations with break dates and attributes for verification
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from datetime import datetime
import colorsys
import time
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
   sys.path.append(module_path)
from ccd_results_utils.segment_identification import generate_date_ranges, ndvi_loss_calculation
from concurrent.futures import ProcessPoolExecutor, as_completed

## SCRIPT CONFIGS ##
##################################

# Set input directory and output files
#input_directory = r"C:\Users\Public\Documents\outputs_ROI\tabular\T29TQG" # UPDATE
#output_raster_file = r"C:\Users\Public\Documents\outputs_ROI\tabular\T29TQG\processed_outputs\rasters\output_raster_ccd.tif" # UPDATE
input_directory = r"H:\new_parquets_2017_2025\tabular\T29TQF" # small tile for tests
input_directory = r"H:\new_parquets_2017_2025\tabular\T29TNE" # UPDATE
output_raster_file = r"H:\new_parquets_2017_2025\tabular\T29TNE\processed_outputs\rasters\output_raster_ccd.tif" # dates will be inserted into the filename based on the date range
output_raster_file=r"C:\Users\mlc\Downloads\TNE_raster_ccd_20_bands.tif"

# Vector file is not set up
output_vector_file = None # Add path if vector file is wanted, to check which points were processed to make the raster

# List of date ranges to filter for, in format (start_date, end_date)
# Use "YYYY-MM-DD" for date values
# Raster will be created for each date range pair

# Example 1 – use a fixed range (no splitting):
date_ranges = generate_date_ranges([("2018-09-30", "2021-12-31")],auto_intervals=False)

# bands names in parquets see https://github.com/S2change/vegetation_loss/tree/main/data_info
# if empty, no additional bands will be added to the output raster (only the 4 main bands: last_tEnd, last_tBreak, is_break, ndvi_last_segment)
start_bands= ["greenStart", "greenStart2", "redStart", "redStart2", "nirStart", "nirStart2" , "swir2Start", "swir2Start2"] # MC additional bands to add to the output raster (values at segment start)
end_bands= ["greenEnd", "greenEnd2", "redEnd", "redEnd2", "nirEnd", "nirEnd2" , "swir2End", "swir2End2"] # MC additional bands to add to the output raster (values at segment end)
nan_tuple = (np.nan,) * (len(end_bands) + len(start_bands)) # tuple of zeros for additional bands in no-break case (i.e. is_break not 1)

SOURCE_CRS = "EPSG:32629" # CRS of input coordinates (UTM zone 29N)
TARGET_CRS = "EPSG:32629" # CRS of input coordinates (UTM zone 29N)

# Example 2 – automatically generate 2-month intervals:
#date_ranges = generate_date_ranges([("2023-01-01", "2024-12-31")], auto_intervals=True, months=2)

# Number of parallel worker processes to use to be determined by number_of_workers() function depending on the system
# num_workers = 22 # NEOUSYS: 12;  DELL: 22 ()

# Boundary shapefile filtering (set to None to disable)
boundary_shapefile = None  # Path to shapefile for spatial boundary filtering

qgis_style_file = True  # Set to True if a .qml style file should be created

# Timer for testing
set_timer = True

# chunk_size for creating the output raster in chunks to avoid memory issues (number of rows to process at a time when creating the output raster)
chunk_size = 500_000
        
################################ number of wortkers depending on the platform (MC feb 2026)

def number_of_workers():
    '''Determine the number of worker processes to use for parallel processing.'''
    num_workers = os.cpu_count() or 1
    return num_workers

print(f"Number of worker processes to be used: {number_of_workers()}"  )

##################################

def calculate_ndvi(input_row):
    """
    Calculate NDVI (Normalized Difference Vegetation Index) from NIR and Red band values.

    Parameters:
    -----------
    input_row : pandas.Series or dict-like
        Row containing 'nirEnd' and 'redEnd' values

    Returns:
    --------
    float : NDVI value calculated as (NIR - Red) / (NIR + Red)
    """
    ndvi = (input_row["nirEnd"] - input_row["redEnd"]) / (input_row["nirEnd"] + input_row["redEnd"])
    return ndvi

def date_conversion_ms(start_date, end_date):
    """
    Convert start and end dates to milliseconds since Unix epoch.

    Parameters:
    -----------
    start_date : str, datetime, or None
        Start date for filtering
    end_date : str, datetime, or None
        End date for filtering

    Returns:
    --------
    tuple : (start_date_ms, end_date_ms) as milliseconds or None
    """
    start_date_ms = None
    end_date_ms = None

    if start_date is not None:
        start_date_dt = pd.to_datetime(start_date)
        start_date_ms = int(start_date_dt.timestamp() * 1000)

    if end_date is not None:
        end_date_dt = pd.to_datetime(end_date)
        end_date_ms = int(end_date_dt.timestamp() * 1000)

    return start_date_ms, end_date_ms

def load_boundary_shapefile(shapefile_path, source_crs=SOURCE_CRS):
    """
    Load boundary shapefile and ensure it's in the same CRS as the data
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the boundary shapefile
    source_crs : str
        CRS of the input data (default: EPSG:32629)
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Boundary geometry in the same CRS as the input data
    """
    try:
        boundary_gdf = gpd.read_file(shapefile_path)
        
        # Reproject to match source CRS if necessary
        if boundary_gdf.crs.to_string() != source_crs:
            print(f"Reprojecting boundary from {boundary_gdf.crs} to {source_crs}")
            boundary_gdf = boundary_gdf.to_crs(source_crs)
        
        # Dissolve all geometries into a single boundary if multiple features exist
        boundary_dissolved = boundary_gdf.dissolve().reset_index(drop=True)
        
        print(f"Loaded boundary shapefile: {shapefile_path}")
        print(f"Boundary CRS: {boundary_dissolved.crs}")
        print(f"Number of boundary features: {len(boundary_gdf)} (dissolved to 1)")
        
        return boundary_dissolved
        
    except Exception as e:
        raise Exception(f"Error loading boundary shapefile {shapefile_path}: {str(e)}")
    
def filter_points_by_boundary(df, boundary_gdf, source_crs=SOURCE_CRS):
    """
    Separate points into those within and outside the boundary.

    Uses spatial join to efficiently filter points by boundary geometry.
    Points outside the boundary are identified and can be marked as no-break pixels.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with x_coord and y_coord columns
    boundary_gdf : geopandas.GeoDataFrame
        Boundary geometry for spatial filtering (dissolved to single polygon)
    source_crs : str
        CRS of the coordinates (default: EPSG:32629)

    Returns:
    --------
    tuple: (points_within_df, unique_pixels_outside)
        - points_within_df: DataFrame with all rows for points inside boundary
        - unique_pixels_outside: DataFrame with unique (x_coord, y_coord) pairs outside boundary
    """
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x_coord, df.y_coord),
        crs=source_crs
    )

    # Within points
    points_within = gpd.sjoin(points_gdf, boundary_gdf, predicate='within', how='inner')
    within_df = points_within.drop(columns=['geometry', 'index_right']).reset_index(drop=True)

    # Outside points
    within_indices = set(points_within.index)
    all_indices = set(points_gdf.index)
    outside_indices = all_indices - within_indices
    if outside_indices:
        outside_df = df.loc[list(outside_indices)]
        unique_pixels_outside = outside_df[['x_coord', 'y_coord']].drop_duplicates().reset_index(drop=True)
    else:
        unique_pixels_outside = pd.DataFrame(columns=['x_coord', 'y_coord'])

    # print(f"  Total segments in parquet: {len(df)}")
    # print(f"  Segments within boundary: {len(within_df)}")
    # print(f"  Unique pixels outside boundary: {len(unique_pixels_outside)}")

    return within_df, unique_pixels_outside
    
def date_filtering(date_value_ms, search_start_ms=None, search_end_ms=None):
    """
    Check if a date value (in milliseconds) falls within the specified date range.

    Parameters:
    -----------
    date_value_ms : int
        The date to check (as milliseconds since Unix epoch)
    search_start_ms : int or None
        Start date for filtering (as milliseconds since Unix epoch)
    search_end_ms : int or None
        End date for filtering (as milliseconds since Unix epoch)

    Returns:
    --------
    bool : True if date passes the filter, False otherwise
    """
    if search_start_ms is None and search_end_ms is None:
        return True

    if date_value_ms is None or pd.isna(date_value_ms):
        return False

    if search_start_ms is not None:
        if date_value_ms < search_start_ms:
            return False

    if search_end_ms is not None:
        if date_value_ms > search_end_ms:
            return False

    return True
    
def process_parquet_file_optimized(file_path, date_ranges_ms_list, boundary_gdf=None, source_crs=SOURCE_CRS):
    """
    Process a single parquet file and return pixel results for MULTIPLE date ranges in a single pass.
    For each pixel and each date range, identifies the most recent break that passes date filtering and NDVI loss validation.

    Algorithm:
    - Iterates through segments in reverse order (newest to oldest) ONCE
    - Groups segments by pixel (x_coord, y_coord)
    - For each pixel and each date range:
        * If the last segment has tBreak != tEnd, returns is_break=-1 (uncertain break)
        * Otherwise, compares consecutive segments using NDVI loss calculation
        * Returns is_break=1 (valid break) if NDVI loss is confirmed
        * Returns is_break=0 (no break) if no valid break is found
    - Only processes segments that pass date filtering for at least one date range

    Parameters:
    -----------
    file_path : str
        Path to the parquet file
    date_ranges_ms_list : list of tuples
        List of (search_start_ms, search_end_ms, date_range_index) tuples
        Each tuple contains start/end dates in milliseconds since Unix epoch and an index
    boundary_gdf : geopandas.GeoDataFrame, optional
        Boundary geometry for spatial filtering. Pixels outside boundary are marked as no-break.
    source_crs : str
        CRS of the coordinates

    Returns:
    --------
    dict : Dictionary mapping date_range_index to list of tuples (x_coord, y_coord, is_break, tEnd_used, tBreak_used, ndvi_last_segment)
        - is_break: 1 (valid break), 0 (no break), or -1 (uncertain break)
        - tEnd_used: tEnd value for breaks (ms since Unix epoch), None for no breaks
        - tBreak_used: tBreak value for breaks (ms since Unix epoch), None for no breaks
        - ndvi_last_segment: NDVI value of the last segment (NaN for no breaks)
    """
    df = pd.read_parquet(file_path)

    # Initialize results dictionary for each date range
    results_by_date_range = {idx: [] for _, _, idx in date_ranges_ms_list}

    # Track which pixels we've fully processed for each date range
    processed_pixels_by_range = {idx: set() for _, _, idx in date_ranges_ms_list}

    # Filter by boundary
    unique_pixels_outside = pd.DataFrame(columns=['x_coord', 'y_coord'])
    if boundary_gdf is not None:
        df, unique_pixels_outside = filter_points_by_boundary(df, boundary_gdf, source_crs)

        # Add unique outside pixels to all date ranges with no-break values
        # Format: (x_coord, y_coord, is_break=0, tEnd_used=None, tBreak_used=None, ndvi_last_segment=np.nan)
        for _, pixel_row in unique_pixels_outside.iterrows():
            x_coord = pixel_row['x_coord']
            y_coord = pixel_row['y_coord']
            for _, _, date_range_idx in date_ranges_ms_list:
                results_by_date_range[date_range_idx].append(
                    (x_coord, y_coord, 0, None, None, np.nan) + nan_tuple # (MC) add dummy values for the additional bands to be added to the output raster (currently set to 0, but can be replaced with actual values if needed
                )

    # Store segments we're currently collecting for a pixel
    current_pixel = None
    current_segments = []

    # Iterate in reverse (newest to oldest)
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        x, y = row["x_coord"], row["y_coord"]
        pixel_key = (x, y)

        # If we've moved to a different pixel, process the previous one for all date ranges
        if current_pixel is not None and pixel_key != current_pixel:
            # Process this pixel for each date range
            for search_start_ms, search_end_ms, date_range_idx in date_ranges_ms_list:
                if current_pixel in processed_pixels_by_range[date_range_idx]:
                    continue

                # Process segments for this date range
                result = process_pixel_segments(current_segments, search_start_ms, search_end_ms)
                results_by_date_range[date_range_idx].append(
                    (current_pixel[0], current_pixel[1]) + result # MC , result[0], result[1], result[2], result[3])
                )
                processed_pixels_by_range[date_range_idx].add(current_pixel)

            # Start collecting for new pixel
            current_pixel = pixel_key
            current_segments = [row]
        elif current_pixel is None:
            # First pixel encountered
            current_pixel = pixel_key
            current_segments = [row]
        else:
            # Same pixel, add segment
            current_segments.append(row)

        # Process last pixel if it exists
    if current_pixel is not None:
        for search_start_ms, search_end_ms, date_range_idx in date_ranges_ms_list:
            if current_pixel not in processed_pixels_by_range[date_range_idx]:
                result = process_pixel_segments(current_segments, search_start_ms, search_end_ms)
                results_by_date_range[date_range_idx].append(
                    (current_pixel[0], current_pixel[1]) + result # MC , result[0], result[1], result[2], result[3])
                )

    return results_by_date_range

def process_pixel_segments(segments, search_start_ms, search_end_ms): # MC where I can get more variables/bands from the segment to add to the output, if needed
    """
    Process a list of segments for a single pixel to determine break status.
    Segments should be in reverse chronological order (newest first).

    This function replicates the original logic: iterate through segments in reverse order,
    check date filtering as we go, and return as soon as a break is found or confirmed.

    Parameters:
    -----------
    segments : list
        List of pandas Series representing segments for a single pixel (newest first)
    search_start_ms : int or None
        Start date for filtering (as milliseconds since Unix epoch)
    search_end_ms : int or None
        End date for filtering (as milliseconds since Unix epoch)

    Returns:
    --------
    tuple : (is_break, tEnd_used, tBreak_used, ndvi_last_segment)
        - is_break: 1 (valid break), 0 (no break), or -1 (uncertain break)
        - tEnd_used: tEnd value for breaks (ms since Unix epoch), None for no breaks
        - tBreak_used: tBreak value for breaks (ms since Unix epoch), None for no breaks
        - ndvi_last_segment: NDVI value of the last segment (NaN for no breaks)
        - additional bands: values for each band in end_bands and start_bands (0 if not available)
    """
    filtered_segments = []
    
    for seg in segments:

        filtered_segments.append(seg)

        date_check = date_filtering(seg["tBreak"], search_start_ms, search_end_ms)
        if not date_check:
            continue

        # Check if we can determine the result early
        if len(filtered_segments) == 1:
            last_seg = filtered_segments[0]
            last_tBreak = last_seg["tBreak"]
            last_tEnd = last_seg["tEnd"]

            if pd.notna(last_tBreak) and last_tBreak != 0 and pd.notna(last_tEnd) and last_tEnd != 0 and last_tBreak != last_tEnd:
                ndvi = calculate_ndvi(last_seg)
                # add to tuple dummy  values for the additional bands to be added to the output raster (currently set to 0, but can be replaced with actual values from the segment if needed)
                return (-1, last_tEnd, last_tBreak, ndvi) + nan_tuple # uncertain break case: tBreak != tEnd but we have no previous segment to compare to, return is_break=-1 with tEnd and tBreak of the last segment, and ndvi of the last segment (the one that ends at the break date)

        # We need at least 2 segments to check NDVI change
        if len(filtered_segments) >= 2:
            active_segment = filtered_segments[-1]  # Older segment (active at time of break)
            newer_segment = filtered_segments[-2]   # Newer segment

            if newer_segment["redStart"] == 0 and newer_segment["redStart2"] == 0 and newer_segment["nirStart"] == 0 and newer_segment["nirStart2"] == 0:
                ndvi = calculate_ndvi(active_segment)
                return (-1, active_segment["tEnd"], active_segment["tBreak"], ndvi) + nan_tuple

            ndvi_check = ndvi_loss_calculation(active_segment, newer_segment)
            if ndvi_check == 1:
                ndvi = calculate_ndvi(active_segment)
                # add end values of active_segment and start value of newer_segment to the tuple to be added to the output raster (currently set to 0, but can be replaced with actual values from the segments if needed)
                end_values = tuple([active_segment.get(band, 0) for band in end_bands]) # default value 0
                start_values = tuple([newer_segment.get(band, 0) for band in start_bands])
                return (1, active_segment["tEnd"], active_segment["tBreak"], ndvi) + end_values + start_values # MC only good case: true NDVI drop confirmed, return is_break=1 with tEnd and tBreak of the active segment (the one that ends at the break date)

    # If we've gone through all segments and found no break, return no break
    return (0, None, None, np.nan) + nan_tuple # no break case: return is_break=0 with None for tEnd and tBreak, and NaN for NDVI, and dummy values for the additional bands to be added to the output raster (currently set to 0, but can be replaced with actual values if needed)

def process_files_chunked(input_dir, date_ranges_list, boundary_shapefile=None, source_crs=SOURCE_CRS, max_workers=number_of_workers):
    """
    Generator that yields processed data from parquet files one at a time to avoid memory issues.
    Processes multiple date ranges in a single pass through each file.
    Converts date filters to milliseconds once before processing begins for efficiency.

    Parameters:
    -----------
    input_dir : str
        Directory containing parquet files
    date_ranges_list : list of tuples
        List of (start_date, end_date) tuples in format 'YYYY-MM-DD' or datetime object
    boundary_shapefile : str, optional
        Path to shapefile for spatial boundary filtering
    source_crs : str
        CRS of the coordinates
    max_workers : int, optional
     Maximum number of parallel worker processes to use. 
     If None, defaults to using all available CPU cores minus one. 

    Yields:
    -------
    dict : Dictionary mapping date_range_index to list of tuples (x_coord, y_coord, is_break, tEnd_used, tBreak_used, ndvi_last_segment)
    """
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    total_files = len(parquet_files)
    print(f"Found {total_files} parquet files to process")

    # Convert date ranges to milliseconds
    date_ranges_ms_list = []
    for idx, (start_date, end_date) in enumerate(date_ranges_list):
        start_ms, end_ms = date_conversion_ms(start_date, end_date)
        date_ranges_ms_list.append((start_ms, end_ms, idx))

    # Load boundary shapefile only once
    boundary_gdf = None
    if boundary_shapefile is not None:
        boundary_gdf = load_boundary_shapefile(boundary_shapefile, source_crs)

    # Diagnostics
    print("Processing multiple date intervals:")
    for idx, (start_date, end_date) in enumerate(date_ranges_list):
        print(f"  - Interval {idx + 1}: {start_date} to {end_date}")

    if boundary_shapefile is not None:
        print(f"Spatial filter: using boundary {boundary_shapefile}")

    # Parallel execution
    results_all = []
    with ProcessPoolExecutor(max_workers=max_workers - 1) as executor:
        futures = {
            executor.submit(
                process_parquet_file_optimized,
                file_path,
                date_ranges_ms_list,
                boundary_gdf,
                source_crs
            ): file_path
            for file_path in parquet_files
        }

        # Progress counter
        completed = 0
        for future in as_completed(futures):
            file_path = futures[future]
            completed += 1
            try:
                result = future.result()
                print(f"[✓] [{completed}/{total_files}] Processed: {os.path.basename(file_path)}")
                results_all.append(result)
            except Exception as e:
                print(f"[X] [{completed}/{total_files}] Error while processing {file_path}: {e}")
                
    # Yield results after all files are processed
    for result_dict in results_all:
        yield result_dict

def collect_pixel_data_chunked(input_dir, date_ranges_list, boundary_shapefile=None, source_crs=SOURCE_CRS):
    """
    Collect and aggregate pixel data from all parquet files into DataFrames for each date range.
    Processes all date ranges in a single pass through each parquet file.

    Parameters:
    -----------
    input_dir : str
        Directory containing parquet files
    date_ranges_list : list of tuples
        List of (start_date, end_date) tuples in format 'YYYY-MM-DD' or datetime object
    boundary_shapefile : str, optional
        Path to shapefile for spatial boundary filtering
    source_crs : str
        CRS of the coordinates

    Returns:
    --------
    dict : Dictionary mapping date_range_index to pandas.DataFrame
        Each DataFrame has columns: x_coord, y_coord, is_break, tEnd_used, tBreak_used, ndvi_last_segment, tEnd_used_yyyymmdd, tBreak_used_yyyymmdd
    """
    # Initialize results dictionary for each date range
    all_results_by_range = {idx: [] for idx in range(len(date_ranges_list))}

    # Process all files and collect results for each date range
    for results_dict in process_files_chunked(input_dir, date_ranges_list, boundary_shapefile, source_crs, max_workers=number_of_workers()):
        for date_range_idx, results_list in results_dict.items():
            all_results_by_range[date_range_idx].extend(results_list)

    # Define column names 
    columns = ["x_coord", "y_coord", "is_break", "tEnd_used", "tBreak_used", "ndvi_last_segment", "tEnd_used_yyyymmdd", "tBreak_used_yyyymmdd"]
    # MC add additional band names to the columns list if they are specified in the start_bands and end_bands lists
    columns = ["x_coord", "y_coord", "is_break", "tEnd_used", "tBreak_used", "ndvi_last_segment"] + end_bands + start_bands #+ ["tEnd_used_yyyymmdd", "tBreak_used_yyyymmdd"]
    # Create DataFrames for each date range
    dataframes_by_range = {}

    for date_range_idx in range(len(date_ranges_list)):
        data_list = all_results_by_range[date_range_idx]
        total_rows = len(data_list)

        print(f"➤ Creating DataFrame for {date_ranges_list[date_range_idx]} with {total_rows:,} rows...")

        # Creation in chuncks
        dfs = []

        for i in range(0, total_rows, chunk_size):
            chunk = pd.DataFrame(data_list[i:i + chunk_size], columns=columns)
            dfs.append(chunk)
            print(f"   ✓ Chunk {i // chunk_size + 1} created ({len(chunk):,} lines)")

        results_df = pd.concat(dfs, ignore_index=True)
        del dfs, data_list  # Free up memory

        # Convert tEnd_used and tBreak_used from milliseconds to pandas Timestamp
        results_df["tEnd_used"] = pd.to_datetime(results_df["tEnd_used"], unit='ms', utc=True, errors='coerce').dt.tz_localize(None)
        results_df["tBreak_used"] = pd.to_datetime(results_df["tBreak_used"], unit='ms', utc=True, errors='coerce').dt.tz_localize(None)

        # Convert to YYYYMMDD integer format
        results_df["tEnd_used_yyyymmdd"] = results_df["tEnd_used"].dt.strftime("%Y%m%d").fillna("0").astype(int)
        results_df["tBreak_used_yyyymmdd"] = results_df["tBreak_used"].dt.strftime("%Y%m%d").fillna("0").astype(int)

        dataframes_by_range[date_range_idx] = results_df

        print(f"Final DataFrame created for {date_ranges_list[date_range_idx]}"
              f"with {len(results_df):,} rows and {len(results_df.columns)} columns.\n")

    return dataframes_by_range

def calculate_raster_parameters_from_pixels(results_df):
    """
    Calculate raster dimensions and resolution from results DataFrame
    with fixed 10x10 meter resolution. Assumes coordinates are pixel centers.
    Considers all pixels regardless of break status to determine the full extent.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: x_coord, y_coord, is_break, tBreak_used, ndvi_last_segment
    """
    if results_df.empty:
        raise ValueError("No coordinate data found in results DataFrame")

    # Extract all coordinates from the DataFrame
    all_x_coords = results_df['x_coord'].tolist()
    all_y_coords = results_df['y_coord'].tolist()

    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)

    # Fixed 10 meter resolution
    res_x = 10.0
    res_y = 10.0

    # Adjust bounds to account for pixel centers (extend by half pixel in each direction)
    min_x_corner = min_x - res_x / 2
    min_y_corner = min_y - res_y / 2
    max_x_corner = max_x + res_x / 2
    max_y_corner = max_y + res_y / 2

    # Calculate dimensions
    width = int(np.ceil((max_x_corner - min_x_corner) / res_x))
    height = int(np.ceil((max_y_corner - min_y_corner) / res_y))

    # Create transform (origin at top-left corner)
    transform = from_origin(min_x_corner, max_y_corner, res_x, res_y)

    return {
        'width': width,
        'height': height,
        'transform': transform,
        'resolution': (res_x, res_y),
        'bounds': (min_x_corner, min_y_corner, max_x_corner, max_y_corner)
    }

def create_raster_array_from_pixels(results_df, raster_params, start_bands, end_bands):
    """
    Below is the docstring of the original version: without the dynamic handling of additional bands, and with the fixed 4-band output 
    (last_tEnd, last_tBreak, is_break, ndvi_last_segment). 
    The new version of the function will handle dynamically the additional bands specified in the start_bands and end_bands lists, 
    and will add them to the output raster after the 4 main bands, in the order they are specified in the lists (first all end_bands, 
    then all start_bands). Note that only ndvi_last_segment is currently calculated in the process_pixel_segments function, 
    so if you want to add more bands to the output raster, you will need to modify the process_pixel_segments function to calculate the values 
    for those bands and include them in the returned tuple, and also modify the collect_pixel_data_chunked function to include those values 
    in the DataFrame that is passed to this function. ndvi_last_segment is rescaled below by 10000 to be stored as an integer in the raster, 
    to preserve precision while keeping the output raster as integer type.
    
    Create a 4-band raster array from results DataFrame with fixed 10m resolution in UTM.
    Assumes coordinates are pixel centers.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: x_coord, y_coord, is_break, tEnd_used, tBreak_used, ndvi_last_segment, tEnd_used_yyyymmdd, tBreak_used_yyyymmdd
        - is_break = 1: valid_break (confirmed vegetation loss)
        - is_break = 0: no_break (no breaks detected)
        - is_break = -1: uncertain_break (potential break but uncertain)
    raster_params : dict
        Raster parameters from calculate_raster_parameters_from_pixels

    Returns:
    --------
    numpy.ndarray
        3D array with shape (4, height, width) containing:
        - Band 1: last_tEnd (int32, YYYYMMDD format, 0 for no break, -9999 for nodata)
        - Band 2: last_tBreak (int32, YYYYMMDD format, 0 for no break, -9999 for nodata)
        - Band 3: is_break (int8, values: 1/0/-1, -99 for nodata)
        - Band 4: ndvi_last_segment (int32, scaled by 10000, -9999 for nodata)

    Notes:
    ------
    Band 4 NDVI values are scaled by 10000 to preserve precision as integers.
    To get original NDVI values, divide by 10000 (e.g., 5432 -> 0.5432).
    """
    width = raster_params['width']
    height = raster_params['height']
    min_x, min_y, max_x, max_y = raster_params['bounds']
    res_x, res_y = raster_params['resolution']

    # 1. Define the dynamic list of extra bands
    # Order: Fixed 4 + any extra start/end bands
    extra_band_names = ["ndvi_last_segment"] + end_bands + start_bands
    
    # 2. Initialize a dictionary of arrays
    # Fixed bands
    bands_dict = {
        'tEnd': np.full((height, width), -9999, dtype=np.int32),
        'tBreak': np.full((height, width), -9999, dtype=np.int32),
        'is_break': np.full((height, width), -99, dtype=np.int8),
    }
    
    # Dynamic bands
    for name in extra_band_names:
        bands_dict[name] = np.full((height, width), -9999, dtype=np.int32)

    # 3. Vectorized Index Calculation (Pre-calculate for all rows at once)
    x_coords = results_df['x_coord'].values
    y_coords = results_df['y_coord'].values
    
    x_idxs = np.round((x_coords - min_x) / res_x - 0.5).astype(int)
    y_idxs = np.round((max_y - y_coords) / res_y - 0.5).astype(int)

    # 4. Fill the arrays
    # We use zip to iterate through the columns we need efficiently
    # Collect all the columns we need to read from the DF
    dynamic_data = {name: results_df[name].values for name in extra_band_names}
    is_break_val = results_df['is_break'].values
    tEnd_val = results_df['tEnd_used_yyyymmdd'].values
    tBreak_val = results_df['tBreak_used_yyyymmdd'].values

    for i in range(len(results_df)):
        xi, yi = x_idxs[i], y_idxs[i]
        
        # Check bounds
        if 0 <= xi < width and 0 <= yi < height:
            curr_break = is_break_val[i]
            
            # Fill Fixed Bands
            bands_dict['is_break'][yi, xi] = curr_break
            if curr_break == 0:
                bands_dict['tEnd'][yi, xi] = 0
                bands_dict['tBreak'][yi, xi] = 0
            else:
                bands_dict['tEnd'][yi, xi] = tEnd_val[i]
                bands_dict['tBreak'][yi, xi] = tBreak_val[i]

            # Fill Dynamic Bands (scaling by 10000)
            for name in extra_band_names:
                val = dynamic_data[name][i]
                if not pd.isna(val):
                    if name == "ndvi_last_segment":
                        bands_dict[name][yi, xi] = int(np.round(val * 10000))
                    else:
                        bands_dict[name][yi, xi] = int(np.round(val))

    # 5. Stack all arrays in order
    # The order will be: tEnd, tBreak, is_break, ndvi_last_segment, then end_bands, then start_bands
    ordered_keys = ['tEnd', 'tBreak', 'is_break'] + extra_band_names
    final_stack = np.stack([bands_dict[key] for key in ordered_keys])

    return final_stack

# (MC feb 2026) Add a function to save the raster array as a GeoTIFF, with proper metadata and reprojecti  on if needed, 
# and with dynamic handling of the additional bands specified in the start_bands and end_bands lists
def save_geotiff(array, output_file, raster_params, end_bands, start_bands, source_crs, target_crs):
    """
    Save a multi-band numpy array as a GeoTIFF, reprojecting if needed.

    Below is the original docstring: Save a 4-band numpy array as a GeoTIFF file, reprojecting to target CRS if needed.

    Parameters:
    -----------
    array : numpy.ndarray
        3D array with shape (4, height, width) containing:
        - Band 1: last_tEnd (int32, YYYYMMDD format, NoData=-9999)
        - Band 2: last_tBreak (int32, YYYYMMDD format, NoData=-9999)
        - Band 3: is_break (int8, values: -1/0/1, NoData=-99)
        - Band 4: ndvi_last_segment (int32, scaled by 10000, NoData=-9999)
    output_file : str
        Path to output GeoTIFF file
    raster_params : dict
        Raster parameters from calculate_raster_parameters_from_pixels
    source_crs : str
        Source coordinate reference system
    target_crs : str
        Target coordinate reference system

    Notes:
    ------
    Band 4 NDVI values are scaled by 10000 to preserve precision as integers.
    To get original NDVI values, divide by 10000 (e.g., 5432 -> 0.5432).
    """

    # 1. Dynamically determine band count and names
    num_bands = array.shape[0]
    fixed_names = ['last_tEnd', 'last_tBreak', 'is_break']
    extra_names = ['ndvi_last_segment'] + end_bands + start_bands
    band_names = fixed_names + extra_names

    # Ensure names match array depth (sanity check)
    if len(band_names) != num_bands:
        raise ValueError(f"Metadata names ({len(band_names)}) don't match array bands ({num_bands})")

    # 2. Setup metadata
    # Note: is_break is technically int8, but for simplicity in a multi-band 
    # Tiff, we use int32 for all to maintain consistency with the dates and scaled NDVI.
    kwargs = {
        'driver': 'GTiff',
        'height': raster_params['height'],
        'width': raster_params['width'],
        'count': num_bands,
        'dtype': rasterio.int32,
        'crs': source_crs,
        'transform': raster_params['transform'],
        'nodata': -9999,
        'compress': 'lzw'  # Added compression because multi-band files get large!
    }

    if source_crs != target_crs:
        from rasterio.io import MemoryFile
        
        with MemoryFile() as memfile:
            with memfile.open(**kwargs) as src:
                # Write data and set names to the memory-buffered source
                for i in range(num_bands):
                    src.write(array[i].astype(np.int32), i + 1)
                    src.set_band_description(i + 1, band_names[i])

                # Calculate reprojection
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                    'crs': target_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })

                with rasterio.open(output_file, 'w', **dst_kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest)
                        dst.set_band_description(i, src.descriptions[i-1])
    else:
        # No reprojection needed
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(num_bands):
                dst.write(array[i].astype(np.int32), i + 1)
                dst.set_band_description(i + 1, band_names[i])

def save_vector_points(results_df, output_file, target_crs=TARGET_CRS, source_crs=SOURCE_CRS):
    """
    Save all points from the results DataFrame as a vector file.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: x_coord, y_coord, is_break, tBreak_used, ndvi_last_segment, tBreak_used_yyyymmdd
    output_file : str
        Path to output vector file
    target_crs : str
        Target coordinate reference system
    source_crs : str
        Source coordinate reference system
    """
    if results_df.empty:
        print("No data to save as vector points")
        return 0

    # Create a copy for the vector output
    vector_df = results_df.copy()

    # Convert tBreak_used (Timestamp) to date string format
    vector_df['tBreak_date'] = vector_df['tBreak_used'].dt.strftime('%Y-%m-%d')

    # Create GeoDataFrame from the results
    gdf = gpd.GeoDataFrame(
        vector_df,
        geometry=gpd.points_from_xy(vector_df.x_coord, vector_df.y_coord),
        crs=source_crs
    )

    # Reproject if necessary
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Save to file
    gdf.to_file(output_file, driver='GPKG')

    return len(gdf)

def create_qgis_style_file_from_pixels(results_df, output_style_file):
    """
    Create a QGIS .qml style file that colors pixels by year with gradient shading by day of year.
    This styles Band 1 (last_tEnd) of the multi-band raster.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns: x_coord, y_coord, is_break, tEnd_used, tBreak_used, ndvi_last_segment, tEnd_used_yyyymmdd, tBreak_used_yyyymmdd
    output_style_file : str
        Path to output .qml style file
    """

    # Get all unique dates from pixels with breaks (is_break == 1 or -1)
    valid_breaks = results_df[results_df['is_break'] != 0]

    if valid_breaks.empty:
        print("No valid breaks found for styling")
        return

    # Extract unique dates (already as Timestamps)
    unique_dates = valid_breaks['tEnd_used_yyyymmdd'].dropna().unique()

    valid_dates = pd.to_datetime(unique_dates, format='%Y%m%d')

    # Group dates by year
    dates_by_year = {}
    for date in valid_dates:
        year = date.year
        date_int = int(date.strftime('%Y%m%d'))
        if year not in dates_by_year:
            dates_by_year[year] = []
        dates_by_year[year].append(date_int)

    # Sort years and create color map
    years = sorted(dates_by_year.keys())
    cmap = plt.get_cmap('tab20', len(years))

    # Create QML content
    qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22.0" minScale="0" maxScale="1e+08" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer opacity="1" type="paletted" band="1">
      <rasterTransparency/>
      <colorPalette>
'''

    # Add color entries for each date, grouped by year with gradient
    for i, year in enumerate(years):
        # Get base color for this year
        base_rgb = cmap(i)[:3]  # RGB values in 0-1 range

        # Convert to HSV for easier manipulation
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)

        # Get unique dates for this year and sort them
        year_dates = sorted(set(dates_by_year[year]))

        for date_value in year_dates:
            # Ensure date_value is an integer
            date_value = int(date_value)

            # Extract day of year (1-365/366)
            date_obj = datetime.strptime(str(date_value), '%Y%m%d')
            day_of_year = date_obj.timetuple().tm_yday

            # Calculate position in year (0 to 1)
            # Account for leap years
            days_in_year = 366 if date_obj.year % 4 == 0 and (date_obj.year % 100 != 0 or date_obj.year % 400 == 0) else 365
            position = (day_of_year - 1) / (days_in_year - 1)

            # Adjust value (brightness) and saturation based on position
            # Early in year: lighter (higher value, lower saturation)
            # Late in year: darker (lower value, higher saturation)
            new_v = 0.9 - (position * 0.4)  # Goes from 0.9 to 0.5
            new_s = s * (0.5 + position * 0.5)  # Goes from 50% to 100% of original saturation

            # Convert back to RGB
            new_rgb = colorsys.hsv_to_rgb(h, new_s, new_v)
            rgb = [int(c * 255) for c in new_rgb]
            color_hex = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

            # Format label to show month-day
            label = date_obj.strftime('%Y-%m-%d')
            qml_content += f'        <paletteEntry value="{date_value}" color="{color_hex}" label="{label}"/>\n'

    # Add entries for no break pixels (value = 0) and nodata
    qml_content += '''        <paletteEntry value="0" color="#808080" label="No Break (Pixels with no detected breaks)"/>
        <paletteEntry value="-9999" color="#000000" label="No Data" alpha="0"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>'''

    # Save style file
    with open(output_style_file, 'w') as f:
        f.write(qml_content)

    print(f"QGIS style file saved to: {output_style_file}")
    print(f"Years in data: {years}")

def process_directory_to_geotiff(input_dir, output_raster_files, output_vector_files, date_ranges_list,
                                source_crs=SOURCE_CRS, target_crs=TARGET_CRS, boundary_shapefile=None, qgis_style_file=False):
    """
    Original docstring (with only 4 bands), now updated to reflect the dynamic handling of additional bands and the fact that we process all date ranges in a single pass through the parquet files:
    Main function to process all parquet files in a directory and save multiple 4-band GeoTIFFs (one for each date range) by reading each parquet file only ONCE.
    Uses UTM coordinates throughout and only reprojects at the end if needed.

    The output GeoTIFF contains 4 bands:
    - Band 1: last_tEnd (YYYYMMDD format, 0 for no break, -9999 for NoData)
    - Band 2: last_tBreak (YYYYMMDD format, 0 for no break, -9999 for NoData)
    - Band 3: is_break (1=valid_break, 0=no_break, -1=uncertain_break, -99=NoData)
    - Band 4: ndvi_last_segment (int32 scaled by 10000, -9999 for NoData)
    - Additional bands can be added dynamically based on the start_bands and end_bands lists, and will be included in the output raster after the 4 main bands.

    Parameters:
    -----------
    input_dir : str
        Directory containing parquet files
    output_raster_files : list of str
        List of paths for output GeoTIFF files (one per date range)
    output_vector_files : list of str or None
        List of paths for output vector files (None elements to skip)
    date_ranges_list : list of tuples
        List of (start_date, end_date) tuples in format 'YYYY-MM-DD' or datetime object
    target_crs : str
        Target coordinate reference system
    boundary_shapefile : str, optional
        Path to shapefile for spatial boundary filtering
    qgis_style_file : bool
        Whether to create a QGIS style file
    """
    # Create output directories if they don't exist
    for output_file in output_raster_files + output_vector_files:
        if output_file is None:
            continue
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect pixel data using chunked processing - PROCESSES ALL DATE RANGES AT ONCE
    print(f"\n{'='*70}")
    print(f"Processing {len(date_ranges_list)} date ranges in a single pass through all parquet files")
    print(f"{'='*70}\n")

    dataframes_by_range = collect_pixel_data_chunked(input_dir, date_ranges_list, boundary_shapefile)

    # Process each date range's results
    for date_range_idx in range(len(date_ranges_list)):
        start_date, end_date = date_ranges_list[date_range_idx]
        output_raster_file = output_raster_files[date_range_idx]
        output_vector_file = output_vector_files[date_range_idx]
        results_df = dataframes_by_range[date_range_idx]

        print(f"\n{'='*70}")
        print(f"Creating outputs for date range {date_range_idx + 1}: {start_date} to {end_date}")
        print(f"Output file: {output_raster_file}")
        print(f"{'='*70}\n")

        if results_df.empty:
            print("No data found for this date range")
            continue

        # Create QGIS style file based on break data
        if qgis_style_file == True and not results_df.empty:
            style_file = output_raster_file.replace('.tif', '.qml')
            create_qgis_style_file_from_pixels(results_df, style_file)

        # Calculate raster parameters from all pixels
        raster_params = calculate_raster_parameters_from_pixels(results_df)

        print(f"Creating raster with dimensions: {raster_params['width']} x {raster_params['height']}")
        print(f"Resolution: {raster_params['resolution'][0]} x {raster_params['resolution'][1]} meters")

        # MC: Create raster array from pixel data, including dynamic handling of additional bands
        raster_array = create_raster_array_from_pixels(results_df, raster_params,start_bands, end_bands) # MC add start_bands and end_bands as parameters to create_raster_array_from_pixels to handle the dynamic additional bands in the output raster

        print('raster array created with shape:', raster_array.shape)

        # Save to GeoTIFF (with optional reprojection)
        save_geotiff(raster_array, output_raster_file, raster_params,  end_bands, start_bands, source_crs, target_crs)

        print(f"4-band GeoTIFF saved to: {output_raster_file}")
        print(f"  - Band 1: last_tEnd (YYYYMMDD format)")
        print(f"  - Band 2: last_tBreak (YYYYMMDD format)")
        print(f"  - Band 3: is_break (1=valid, 0=none, -1=uncertain)")
        print(f"  - Band 4: ndvi_last_segment (scaled by 10000)")

        # Save vector points if requested
        if output_vector_file is not None:
            num_points_saved = save_vector_points(results_df, output_vector_file, target_crs, source_crs)
            print(f"Vector points saved to: {output_vector_file}")
            print(f"Points saved to vector file: {num_points_saved}")

        # Summary statistics
        total_pixels = len(results_df)
        valid_breaks = len(results_df[results_df['is_break'] == 1])
        no_breaks = len(results_df[results_df['is_break'] == 0])
        uncertain_breaks = len(results_df[results_df['is_break'] == -1])

        print(f"\nTotal pixels processed: {total_pixels}")
        print(f"  - Pixels with valid breaks (is_break=1): {valid_breaks}")
        print(f"  - Pixels with no breaks (is_break=0): {no_breaks}")
        print(f"  - Pixels with uncertain breaks (is_break=-1): {uncertain_breaks}")
        print(f"  - Pixels not in parquet files will show as NoData")

if __name__ == "__main__":
    if set_timer == True:
        start_time = time.time()
        print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

    # Prepare output file lists for all date ranges
    output_raster_files = []
    output_vector_files = []

    for start_date, end_date in date_ranges:
        # Create unique filenames for each date range
        # Convert dates to string format for filename (YYYYMMDD)
        start_str = start_date.replace("-", "") if start_date else "NoStart"
        end_str = end_date.replace("-", "") if end_date else "NoEnd"
        date_suffix = f"_{start_str}_to_{end_str}"

        # Insert date suffix before file extension
        base_raster_file = output_raster_file.replace('.tif', f'{date_suffix}.tif')
        output_raster_files.append(base_raster_file)

        # Handle vector file if specified
        if output_vector_file is not None:
            base_vector_file = output_vector_file.replace('.gpkg', f'{date_suffix}.gpkg')
        else:
            base_vector_file = None
        output_vector_files.append(base_vector_file)

    # Process all date ranges in a single pass through the parquet files
    process_directory_to_geotiff(
        input_directory,
        output_raster_files,
        output_vector_files,
        date_ranges,
        boundary_shapefile=boundary_shapefile,
        qgis_style_file=qgis_style_file
    ) # target_crs='EPSG:4326'

    if set_timer == True:
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("="*70)
        print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

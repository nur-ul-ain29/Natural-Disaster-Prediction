"""
gee_pipeline.py
================
Google Earth Engine integration -- five things this adds:

  1. TIME-SERIES ANALYSIS (technical task #1): pull a sequence of
     satellite images over the same location across many dates, and
     track a trend (here, NDWI -- water presence) over time.

  2. LIVE INFERENCE: pull a real before/after Sentinel-1 pair for any
     location/date and feed it through your trained flood model.

  3. Download a real Sentinel-2 image as a local GeoTIFF (for the
     wildfire detector, which needs actual files).

  4. Live, single-reading NDWI (water) for the flood prediction model.

  5. Live, single-reading NDMI (vegetation moisture) for the wildfire
     prediction model's optional live-signal check.

SETUP (run once per Colab session):
    !pip install earthengine-api geemap
    import ee
    ee.Authenticate()
    ee.Initialize(project='YOUR-GCP-PROJECT-ID')
"""

import numpy as np


def _require_ee(project_id=None):
    """
    Import ee and actually initialize it using already-saved credentials
    from a prior ee.Authenticate() call. This matters because every
    `!python script.py` run in Colab starts a fresh process -- saved
    login credentials persist on disk, but ee.Initialize() still has to
    be called explicitly in THIS process, or every ee.* call fails with
    "client library not initialized" even though you're already logged in.
    """
    import os
    project_id = project_id or os.environ.get("GEE_PROJECT_ID", "disaster-project-508518")
    try:
        import ee
        ee.Initialize(project=project_id)
        ee.Number(1).getInfo()  # cheap call confirming it actually works
        return ee
    except Exception as e:
        raise RuntimeError(
            "Google Earth Engine isn't set up in this session. If you've "
            "never run ee.Authenticate() at all, run in a notebook cell:\n"
            "  !pip install earthengine-api geemap\n"
            "  import ee; ee.Authenticate(); ee.Initialize(project='YOUR-PROJECT-ID')\n"
            "If you HAVE already authenticated before, this may be a project-ID "
            f"mismatch -- this script defaults to project_id='{project_id}'; "
            "set the GEE_PROJECT_ID environment variable if yours differs.\n"
            f"Original error: {e}"
        )


def get_ndwi_time_series(lat, lon, days_back=60, buffer_meters=500):
    """
    Time-series analysis (technical task #1): NDWI computed from
    Sentinel-2 for a point location, across every cloud-free image in
    the last `days_back` days. Returns (dates, ndwi_values).
    """
    ee = _require_ee()
    from datetime import datetime, timedelta

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def add_ndwi(img):
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
        return img.addBands(ndwi)

    with_ndwi = collection.map(add_ndwi)
    series = with_ndwi.select("NDWI").getRegion(point.centroid(), scale=10).getInfo()

    header, rows = series[0], series[1:]
    date_idx = header.index("time")
    ndwi_idx = header.index("NDWI")

    dates, values = [], []
    for row in rows:
        if row[ndwi_idx] is not None:
            dates.append(datetime.utcfromtimestamp(row[date_idx] / 1000).strftime("%Y-%m-%d"))
            values.append(row[ndwi_idx])

    return dates, values


def get_sentinel1_pair(lat, lon, before_date, after_date, buffer_meters=2000, window_days=10):
    """Live before/after Sentinel-1 VV/VH pair as numpy arrays, shape (2, H, W) each."""
    ee = _require_ee()
    from datetime import datetime, timedelta

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)

    def closest_image(target_date_str):
        target = datetime.strptime(target_date_str, "%Y-%m-%d")
        window_start = (target - timedelta(days=window_days)).strftime("%Y-%m-%d")
        window_end = (target + timedelta(days=window_days)).strftime("%Y-%m-%d")
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(point)
            .filterDate(window_start, window_end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        return collection.select(["VV", "VH"]).first()

    def image_to_array(img):
        region = img.clip(point).sampleRectangle(region=point, defaultValue=0)
        vv = np.array(region.get("VV").getInfo())
        vh = np.array(region.get("VH").getInfo())
        return np.stack([vv, vh], axis=0).astype(np.float32)

    before_img = closest_image(before_date)
    after_img = closest_image(after_date)
    return image_to_array(before_img), image_to_array(after_img)


def download_sentinel2_geotiff(lat, lon, date, out_path, bands=("B8", "B4", "B12"),
                                buffer_meters=3000, window_days=15):
    """Downloads a real Sentinel-2 image as a local GeoTIFF (NIR, Red, SWIR bands)."""
    ee = _require_ee()
    import requests
    from datetime import datetime, timedelta

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    target = datetime.strptime(date, "%Y-%m-%d")
    window_start = (target - timedelta(days=window_days)).strftime("%Y-%m-%d")
    window_end = (target + timedelta(days=window_days)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(window_start, window_end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    image = collection.select(list(bands)).first().clip(point)

    url = image.getDownloadURL({"scale": 20, "region": point, "format": "GEO_TIFF"})
    response = requests.get(url)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"Saved: {out_path} (bands: {bands})")
    return out_path


def get_current_ndwi(lat, lon, window_days=15, buffer_meters=500):
    """Live, present-day NDWI reading for a single point -- one value, not a trend."""
    ee = _require_ee()
    from datetime import datetime, timedelta

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    end = datetime.utcnow()
    start = end - timedelta(days=window_days)

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )
    ndwi = image.normalizedDifference(["B3", "B8"])
    value = ndwi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).get("nd").getInfo()
    return value


def get_current_ndmi(lat, lon, window_days=15, buffer_meters=500):
    """Live, present-day NDMI (vegetation moisture) reading for a single point."""
    ee = _require_ee()
    from datetime import datetime, timedelta

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    end = datetime.utcnow()
    start = end - timedelta(days=window_days)

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )
    ndmi = image.normalizedDifference(["B8", "B11"])
    value = ndmi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=20).get("nd").getInfo()
    return value


def get_ndwi_before_date(lat, lon, reference_date, days_back=30, buffer_meters=500):
    """
    Real precursor signal: NDWI trend for the `days_back` days BEFORE a
    given reference date (not "now") -- this is what makes a genuine
    before-event forecast possible, unlike get_current_ndwi (which only
    ever looks at today). Used with real historical flood event dates
    from GLOBAL_FLOOD_DB/MODIS_EVENTS/V1.

    Returns (mean_ndwi, trend_slope) or (None, None) if no cloud-free
    imagery was available in the window. trend_slope is the linear
    slope of NDWI over the window (positive = rising water in the
    days leading up to reference_date, a real flood precursor signal).
    """
    ee = _require_ee()
    from datetime import datetime, timedelta

    ref = datetime.strptime(reference_date, "%Y-%m-%d") if isinstance(reference_date, str) else reference_date
    start = ref - timedelta(days=days_back)

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start.strftime("%Y-%m-%d"), ref.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    )

    # Guard against an empty collection (no cloud-free imagery in this
    # window/location at all) -- selecting bands on an empty collection
    # raises "No bands in collection" instead of just having no data.
    if collection.size().getInfo() == 0:
        return None, None

    def add_ndwi(img):
        return img.addBands(img.normalizedDifference(["B3", "B8"]).rename("NDWI"))

    series = (
        collection.map(add_ndwi).select("NDWI")
        .getRegion(point.centroid(), scale=10).getInfo()
    )
    header, rows = series[0], series[1:]
    t_idx, v_idx = header.index("time"), header.index("NDWI")

    times, values = [], []
    for row in rows:
        if row[v_idx] is not None:
            times.append(row[t_idx])
            values.append(row[v_idx])

    if len(values) < 2:
        return None, None

    # De-duplicate same-day observations (Sentinel-2 sometimes captures
    # the same area twice in one day from overlapping orbit paths) --
    # without this, two near-identical timestamps with a value difference
    # divided by a near-zero time difference produces wildly unstable
    # slope estimates (seen in practice: values like +79 or -136, which
    # are meaningless for an index that only ranges roughly -1 to 1).
    day_buckets = {}
    for t, v in zip(times, values):
        day = int(t // 86400000)  # ms -> whole day bucket
        day_buckets.setdefault(day, []).append(v)
    days_sorted = sorted(day_buckets)
    dedup_values = np.array([np.mean(day_buckets[d]) for d in days_sorted])
    dedup_days = np.array([d - days_sorted[0] for d in days_sorted], dtype=float)

    mean_ndwi = float(dedup_values.mean())
    if len(dedup_days) < 2 or (dedup_days[-1] - dedup_days[0]) < 3:
        # Not enough real time spread to fit a meaningful trend -- report
        # the mean only, slope 0, rather than an unstable/meaningless number.
        slope = 0.0
    else:
        slope = float(np.polyfit(dedup_days, dedup_values, 1)[0])
    return mean_ndwi, slope



    import json

    dates, values = get_ndwi_time_series(31.5497, 74.3436, days_back=60)
    print("Date        NDWI")
    for d, v in zip(dates, values):
        print(f"{d}  {v:.3f}")

    # Save real results to a file, so report charts can be regenerated
    # from actual saved data instead of transcribed numbers.
    with open("results_ndwi_timeseries.json", "w") as f:
        json.dump({"dates": dates, "values": values}, f, indent=2)
    print("\nSaved: results_ndwi_timeseries.json")
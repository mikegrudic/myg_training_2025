import datetime
from dataclasses import dataclass
from functools import cache
from xml.etree import ElementTree

import numpy as np

GAP_TABLE_PATH = "strava_GAP_table.dat"
EARTH_RADIUS_M = 6371000.0


@cache
def _load_gap_table():
    """Load and cache the Strava grade-vs-GAP-factor table from disk.

    Returns
    -------
    (grade, factor) : tuple of np.ndarray
        ``grade`` is the slope in percent (assumed monotonically
        increasing); ``factor`` is the corresponding pace-adjustment
        multiplier used by Strava's grade-adjusted-pace model.
    """
    grade, factor = np.loadtxt(GAP_TABLE_PATH).T
    return grade, factor


def _strava_gap_factor(grade_percent):
    """Strava GAP factor at one or more grade values, by interpolating the
    tabulated Strava grade-vs-factor curve and linearly extrapolating
    beyond its endpoints."""
    grade_table, factor_table = _load_gap_table()
    grade_percent = np.asarray(grade_percent, dtype=float)
    factor = np.interp(grade_percent, grade_table, factor_table)

    below = grade_percent < grade_table[0]
    if np.any(below):
        slope = (factor_table[1] - factor_table[0]) / (grade_table[1] - grade_table[0])
        factor[below] = factor_table[0] + slope * (grade_percent[below] - grade_table[0])

    above = grade_percent > grade_table[-1]
    if np.any(above):
        slope = (factor_table[-1] - factor_table[-2]) / (grade_table[-1] - grade_table[-2])
        factor[above] = factor_table[-1] + slope * (grade_percent[above] - grade_table[-1])

    return factor


def _minetti_gap_factor(grade_percent):
    """Minetti et al. (2002) energetic cost of running on graded terrain,
    normalized to the flat-ground cost. Polynomial fit:

        C_r(i) = 155.4 i^5 - 30.4 i^4 - 43.3 i^3 + 46.3 i^2 + 19.5 i + 3.6

    where ``i`` is the dimensionless grade (rise/run) and ``C_r(0) = 3.6``.
    Validated over i in [-0.45, +0.45]; extrapolates polynomially outside.
    """
    i = np.asarray(grade_percent, dtype=float) / 100.0
    c_r = 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6
    return c_r / 3.6


def _gap_factor(grade_percent, model="strava"):
    """Dispatch a grade-adjustment model. ``model`` is ``"strava"`` (default)
    or ``"minetti"``."""
    if model == "minetti":
        return _minetti_gap_factor(grade_percent)
    return _strava_gap_factor(grade_percent)


def _iir_smooth(values, step, length):
    """Single-pole IIR low-pass with a distance-aware time constant.

    For each sample ``i`` the update is
    ``y_i = a_i * x_i + (1 - a_i) * y_{i-1}`` with
    ``a_i = 1 - exp(-step_i / length)``. This makes the effective
    e-folding decay equal to ``length`` in the units of ``step``,
    regardless of non-uniform spacing between samples.

    Parameters
    ----------
    values : np.ndarray
        Series to smooth, length ``N``.
    step : np.ndarray
        Inter-sample step sizes, length ``N - 1``.
    length : float
        E-folding decay length, in the same units as ``step``.
    """
    alpha = 1.0 - np.exp(-step / length)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha[i - 1] * values[i] + (1.0 - alpha[i - 1]) * out[i - 1]
    return out


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters) between paired lat/lon points.

    Inputs are in degrees and may be scalars or arrays. Earth is
    treated as a sphere of radius ``EARTH_RADIUS_M``.
    """
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def parse_waypoints(path):
    """Extract named ``<wpt>`` elements from a GPX file.

    Returns a list of ``(latitude, longitude, name)`` tuples for every
    top-level waypoint that has a ``<name>`` child with text. Waypoints
    without names are skipped.
    """
    tree = ElementTree.parse(path)
    root = tree.getroot()
    ns_uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")] if root.tag.startswith("{") else ""
    ns = {"gpx": ns_uri} if ns_uri else {}
    prefix = "gpx:" if ns else ""
    result = []
    for wpt in root.iterfind(f".//{prefix}wpt", ns):
        name_el = wpt.find(f"{prefix}name", ns)
        if name_el is None or name_el.text is None:
            continue
        name = name_el.text.strip()
        if not name:
            continue
        result.append((float(wpt.attrib["lat"]), float(wpt.attrib["lon"]), name))
    return result


def _parse_gpx_points(path):
    """Extract (lat, lon, elevation) arrays from every ``<trkpt>`` in a GPX file.

    Auto-detects the GPX XML namespace from the document root. Track
    points missing an ``<ele>`` child are skipped. Returns three float
    arrays in track order.
    """
    tree = ElementTree.parse(path)
    root = tree.getroot()
    ns_uri = root.tag[root.tag.find("{") + 1 : root.tag.find("}")] if root.tag.startswith("{") else ""
    ns = {"gpx": ns_uri} if ns_uri else {}
    prefix = "gpx:" if ns else ""

    lats, lons, eles, times = [], [], [], []
    for pt in root.iterfind(f".//{prefix}trkpt", ns):
        ele_el = pt.find(f"{prefix}ele", ns)
        if ele_el is None or ele_el.text is None:
            continue
        lats.append(float(pt.attrib["lat"]))
        lons.append(float(pt.attrib["lon"]))
        eles.append(float(ele_el.text))
        time_el = pt.find(f"{prefix}time", ns)
        if time_el is not None and time_el.text is not None:
            try:
                t = datetime.datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
                times.append(t.timestamp())
            except ValueError:
                times.append(np.nan)
        else:
            times.append(np.nan)
    return np.array(lats), np.array(lons), np.array(eles), np.array(times)


@dataclass
class ElevationProfile:
    """Per-point elevation profile along a track.

    Attributes
    ----------
    distance : np.ndarray
        Cumulative along-track distance (meters), starting at 0.
    elevation : np.ndarray
        Elevation (meters) at each point. May be smoothed; see
        :meth:`from_gpx`.
    grade : np.ndarray
        Local slope (percent), computed via ``np.gradient`` of
        ``elevation`` against ``distance``.
    gap_factor : np.ndarray
        Strava grade-adjusted-pace multiplier at each point. Greater
        than 1 means the segment costs more per meter than flat ground;
        less than 1 means a downhill assist.
    """

    distance: np.ndarray
    elevation: np.ndarray
    grade: np.ndarray
    gap_factor: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    time_seconds: np.ndarray

    @classmethod
    def from_gpx(cls, path, smoothing_length=50.0, model="strava", reverse=False):
        """Build an ElevationProfile from a GPX track.

        Parses ``<trkpt>`` elements (with ``<ele>`` children), computes
        cumulative distance along the track using the haversine formula,
        optionally smooths the elevation series with a single-pole IIR
        filter, then derives grade (in percent) and a grade-adjustment
        factor at each point.

        Parameters
        ----------
        path : str
            Path to the GPX file.
        smoothing_length : float, optional
            E-folding length (meters) for IIR smoothing applied to the
            elevation series before grade is computed. Defaults to ``50``;
            set to ``0`` to disable. Larger values suppress more
            high-frequency altimeter/GPS noise at the cost of phase lag
            and reduced grade resolution.
        model : str, optional
            Grade-adjustment model: ``"strava"`` (default) uses the
            tabulated Strava GAP curve; ``"minetti"`` uses the Minetti
            et al. (2002) polynomial fit.
        reverse : bool, optional
            If True, reverse the trackpoint order so the route is
            traversed end-to-start. Time data is discarded since it would
            be meaningless for the reversed direction.
        """
        lat, lon, elevation, time_abs = _parse_gpx_points(path)
        if reverse:
            lat = lat[::-1]
            lon = lon[::-1]
            elevation = elevation[::-1]
            time_abs = np.full_like(time_abs, np.nan)
        if len(lat) < 2:
            raise ValueError(f"GPX file {path} contains fewer than 2 track points with elevation")

        segment_lengths = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        keep = np.concatenate(([True], segment_lengths > 0))
        lat, lon, elevation, time_abs = lat[keep], lon[keep], elevation[keep], time_abs[keep]
        segment_lengths = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))

        if smoothing_length > 0:
            elevation = _iir_smooth(elevation, segment_lengths, smoothing_length)

        if np.any(np.isfinite(time_abs)):
            time_seconds = time_abs - np.nanmin(time_abs)
        else:
            time_seconds = time_abs

        grade = np.gradient(elevation, distance) * 100.0
        gap_factor = _gap_factor(grade, model=model)
        return cls(
            distance=distance,
            elevation=elevation,
            grade=grade,
            gap_factor=gap_factor,
            latitude=lat,
            longitude=lon,
            time_seconds=time_seconds,
        )
    
    def constant_gap_split_time(self, total_time_minutes, fatigue_rate_per_hour=0.0):
        """Elapsed time (minutes) at each track point.

        Without fatigue (``fatigue_rate_per_hour=0``), the runner holds a
        constant grade-adjusted pace and elapsed time scales linearly with
        cumulative effort.

        With ``fatigue_rate_per_hour > 0``, GAP pace inflates exponentially
        with elapsed time: ``gap_pace(t) = gap_pace(0) * (1 + f)**t`` where
        ``f`` is the fractional decay per hour (e.g. ``0.05`` for 5 %/h).
        ``total_time_minutes`` is conserved by construction.

        Parameters
        ----------
        total_time_minutes : float
            Target total time for the route, in minutes.
        fatigue_rate_per_hour : float, optional
            Fractional GAP-pace inflation per hour of elapsed time. ``0``
            (the default) means no fatigue.
        """
        avg_factor = 0.5 * (self.gap_factor[:-1] + self.gap_factor[1:])
        segment_effort = avg_factor * np.diff(self.distance)
        cumulative_effort = np.concatenate(([0.0], np.cumsum(segment_effort)))
        g_ratio = cumulative_effort / cumulative_effort[-1]

        if fatigue_rate_per_hour == 0.0:
            return total_time_minutes * g_ratio

        k = np.log1p(fatigue_rate_per_hour)
        total_hours = total_time_minutes / 60.0
        decay = 1.0 - np.exp(-k * total_hours)
        return -np.log1p(-decay * g_ratio) / k * 60.0
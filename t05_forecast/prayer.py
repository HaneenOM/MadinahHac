"""Prayer times for Madinah computed from the sun's position (no website, no API).

Standard solar-position formulas (NOAA / Meeus), Umm al-Qura convention:
Fajr when the sun is 18.5 degrees below the horizon, Asr when an object's shadow
equals its length plus its noon shadow, Maghrib at sunset, Isha 90 minutes after
Maghrib. Coordinates of Madinah: 24.4672 N, 39.6111 E, clock UTC+3.

Prayer times are known in advance, so they may be used for forecast holes.
"""
from __future__ import annotations

import datetime as dt
import math

LAT, LON, TZ = 24.4672, 39.6111, 3.0
PRAYERS = ("dhuhr", "asr", "maghrib", "isha")


def _sun(day: dt.date) -> tuple:
    """Solar declination (degrees) and equation of time (minutes) at local noon."""
    jd = day.toordinal() + 1721424.5 + (12.0 - TZ) / 24.0
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    y = math.tan(math.radians(eps / 2.0)) ** 2
    l0r = math.radians(l0)
    eot = 4.0 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                             - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    return decl, eot


def _hour_angle(altitude_deg: float, decl: float) -> float:
    """Minutes between solar noon and the moment the sun is at this altitude."""
    lat, d, a = math.radians(LAT), math.radians(decl), math.radians(altitude_deg)
    cos_h = (math.sin(a) - math.sin(lat) * math.sin(d)) / (math.cos(lat) * math.cos(d))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_h)))) * 4.0


def prayer_minutes(day: str | dt.date) -> dict:
    """Minutes after local midnight of Fajr, Dhuhr, Asr, Maghrib and Isha on this date."""
    if isinstance(day, str):
        day = dt.date.fromisoformat(day)
    decl, eot = _sun(day)
    noon = 720.0 - 4.0 * LON - eot + 60.0 * TZ
    asr_alt = math.degrees(math.atan(1.0 / (1.0 + math.tan(math.radians(abs(LAT - decl))))))
    maghrib = noon + _hour_angle(-0.833, decl)
    return {
        "fajr": noon - _hour_angle(-18.5, decl),
        "dhuhr": noon,
        "asr": noon + _hour_angle(asr_alt, decl),
        "maghrib": maghrib,
        "isha": maghrib + 90.0,
    }


def warp_minutes(minutes, target_day: str, source_day: str):
    """Map clock minutes on the target day to the minutes on the source day that sit
    at the same distance from the prayers (piecewise linear between the four prayers)."""
    import numpy as np
    tp, sp = prayer_minutes(target_day), prayer_minutes(source_day)
    anchors = [tp[p] for p in PRAYERS]
    shifts = [sp[p] - tp[p] for p in PRAYERS]
    minutes = np.asarray(minutes, dtype=float)
    return minutes + np.interp(minutes, anchors, shifts)


if __name__ == "__main__":
    def fmt(x: float) -> str:
        x = int(round(x))
        return f"{x // 60:02d}:{x % 60:02d}"
    for d in ("2026-09-04", "2026-09-05", "2026-09-11", "2026-09-12", "2026-09-18", "2026-09-19"):
        p = prayer_minutes(d)
        print(d, " ".join(f"{k} {fmt(v)}" for k, v in p.items()))

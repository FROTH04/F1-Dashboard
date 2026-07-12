"""
FastF1 Data Loader — echte API statt JSON
Lädt Session-Daten via session.load() und normalisiert sie
auf das Format das die Dashboard-Charts erwarten.
"""

import fastf1
import fastf1.ergast
import pandas as pd
import numpy as np
import os, logging, warnings
from datetime import datetime
from functools import lru_cache

warnings.filterwarnings('ignore')
fastf1.set_log_level('WARNING')

# Cache-Ordner anlegen (spart Bandbreite — jede Session ~200-500 MB)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Team / Fahrer Stammdaten ──────────────────────────────────────────────────
DRIVER_COLORS = {
    'NOR': '#FF8000', 'PIA': '#FF8000',
    'VER': '#3671C6', 'LAW': '#3671C6', 'PER': '#3671C6',
    'LEC': '#E8002D', 'SAI': '#E8002D', 'HAM': '#E8002D',
    'RUS': '#27F4D2', 'ANT': '#27F4D2', 'BOT': '#27F4D2',
    'ALO': '#358C75', 'STR': '#358C75',
    'GAS': '#0093CC', 'DOO': '#0093CC', 'OCO': '#0093CC',
    'ALB': '#64C4FF', 'COL': '#64C4FF',
    'HUL': '#52E252', 'BOR': '#52E252',
    'TSU': '#6692FF', 'HAD': '#6692FF', 'RIC': '#6692FF',
    'MAG': '#B6BABD', 'BEA': '#B6BABD',
    'ZHO': '#52E252', 'SAR': '#64C4FF',
}

TEAM_COLORS = {
    'McLaren': '#FF8000', 'Ferrari': '#E8002D',
    'Red Bull Racing': '#3671C6', 'Red Bull': '#3671C6',
    'Mercedes': '#27F4D2', 'Aston Martin': '#358C75',
    'Alpine': '#0093CC', 'Williams': '#64C4FF',
    'Kick Sauber': '#52E252', 'Sauber': '#52E252',
    'RB': '#6692FF', 'Racing Bulls': '#6692FF',
    'Haas F1 Team': '#B6BABD', 'Haas': '#B6BABD',
}

def get_driver_color(code):
    return DRIVER_COLORS.get(code, '#888888')

def get_team_color(team):
    return TEAM_COLORS.get(team, '#888888')

def td_to_seconds(td):
    """Timedelta / NaT → float Sekunden"""
    if pd.isna(td): return None
    if hasattr(td, 'total_seconds'): return td.total_seconds()
    return float(td)

# ── Schedule ──────────────────────────────────────────────────────────────────

def get_schedule(year):
    """Liefert [{round, name, circuit, country, date, laps}]"""
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        races = []
        today = pd.Timestamp.now(tz='UTC').tz_localize(None)
        for _, ev in sched.iterrows():
            ev_date = pd.Timestamp(ev['EventDate'])
            if ev_date.tzinfo: ev_date = ev_date.tz_localize(None)
            races.append({
                'round':   int(ev['RoundNumber']),
                'name':    ev['EventName'],
                'circuit': ev.get('CircuitShortName', ev['EventName'].replace(' Grand Prix', '')),
                'country': ev['Country'],
                'date':    ev_date.strftime('%Y-%m-%d'),
                'completed': ev_date < today,
            })
        return sorted(races, key=lambda x: x['round'])
    except Exception as e:
        logging.warning(f"Schedule error: {e}")
        return []

# ── Session laden ─────────────────────────────────────────────────────────────

_session_cache = {}

def load_session(year, round_num, session_type='R', telemetry=True):
    """
    Lädt eine FastF1-Session und gibt das Session-Objekt zurück.
    Gecacht pro (year, round, type).
    """
    key = (year, round_num, session_type)
    if key in _session_cache:
        return _session_cache[key]
    
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(
            laps=True,
            telemetry=telemetry,
            weather=True,
            messages=False,
        )
        _session_cache[key] = session
        return session
    except Exception as e:
        logging.warning(f"Session load error ({year} R{round_num} {session_type}): {e}")
        return None

# ── Ergebnisse ────────────────────────────────────────────────────────────────

def get_race_results(session):
    """
    Gibt [{position, code, name, team, color, points, gap, fastest_lap,
           sectors, pit_stops, tyre_strategy}] zurück.
    """
    if session is None or len(session.results) == 0:
        return []

    POINTS_MAP = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    results = []
    laps = session.laps

    for _, row in session.results.iterrows():
        code = str(row.get('Abbreviation', ''))
        pos  = int(row.get('Position', 99)) if not pd.isna(row.get('Position')) else 99

        # Gap (GridPosition nur Platzhalter wenn kein GapToLeader)
        gap = 0.0
        if pos > 1:
            gap_raw = row.get('Time') or row.get('GapToLeader')
            gap = td_to_seconds(gap_raw) or 0.0

        # Schnellste Runde
        driver_laps = laps.pick_driver(code) if len(laps) > 0 else pd.DataFrame()
        fastest_lap_time = None
        sectors = {'S1': None, 'S2': None, 'S3': None}
        if len(driver_laps) > 0:
            try:
                fl = driver_laps.pick_fastest()
                fastest_lap_time = td_to_seconds(fl.get('LapTime'))
                sectors['S1'] = td_to_seconds(fl.get('Sector1Time'))
                sectors['S2'] = td_to_seconds(fl.get('Sector2Time'))
                sectors['S3'] = td_to_seconds(fl.get('Sector3Time'))
            except Exception:
                pass

        # Tyre-Strategie rekonstruieren
        strategy = _get_tyre_strategy(driver_laps)

        results.append({
            'position':     pos,
            'code':         code,
            'name':         f"{row.get('FirstName','')} {row.get('LastName','')}".strip(),
            'team':         str(row.get('TeamName', '')),
            'color':        get_driver_color(code),
            'points':       POINTS_MAP.get(pos, 0),
            'gap':          round(gap, 3),
            'fastest_lap':  fastest_lap_time or 90.0,
            'sectors':      sectors,
            'pit_stops':    _count_pitstops(driver_laps),
            'tyre_strategy': strategy,
        })

    return sorted(results, key=lambda x: x['position'])


def _get_tyre_strategy(driver_laps):
    if len(driver_laps) == 0: return 'M-H'
    try:
        compounds = driver_laps['Compound'].dropna().unique().tolist()
        # Ordnung nach erstem Auftauchen
        seen, ordered = set(), []
        for c in driver_laps['Compound'].dropna():
            if c not in seen:
                seen.add(c)
                ordered.append(c[0] if len(c) > 0 else '?')  # S, M, H, I, W
        return '-'.join(ordered) if ordered else 'M-H'
    except Exception:
        return 'M-H'


def _count_pitstops(driver_laps):
    if len(driver_laps) == 0: return 1
    try:
        return int(driver_laps['PitOutTime'].notna().sum())
    except Exception:
        return 1


# ── Lap Times ─────────────────────────────────────────────────────────────────

def get_lap_times(session, drivers=None):
    """
    Gibt {driver_code: [float, ...]} zurück — Sekunden pro Runde.
    Outlier > 120s gefiltert (Safety Car, VSC, In-laps etc.)
    """
    if session is None or len(session.laps) == 0:
        return {}
    
    laps = session.laps
    result = {}
    all_drivers = drivers or laps['Driver'].unique().tolist()

    for code in all_drivers:
        dl = laps.pick_driver(code)
        times = []
        for _, lap in dl.iterrows():
            t = td_to_seconds(lap.get('LapTime'))
            if t and 60 < t < 200:   # sinnvoller Bereich
                times.append(round(t, 3))
        if times:
            result[code] = times

    return result


# ── Telemetrie ────────────────────────────────────────────────────────────────

def get_telemetry(session, driver_code):
    """
    Liefert {distance, speed, throttle, brake, gear, drs}
    für die schnellste Runde eines Fahrers.
    """
    if session is None or len(session.laps) == 0:
        return None
    try:
        dl = session.laps.pick_driver(driver_code)
        if len(dl) == 0: return None
        fl = dl.pick_fastest()
        tel = fl.get_telemetry().add_distance()
        return {
            'distance': tel['Distance'].round(1).tolist(),
            'speed':    tel['Speed'].tolist(),
            'throttle': tel['Throttle'].tolist(),
            'brake':    (tel['Brake'].astype(float) * 100).tolist(),  # bool → 0/100
            'gear':     tel['nGear'].tolist(),
            'drs':      tel['DRS'].tolist(),
        }
    except Exception as e:
        logging.debug(f"Telemetry error {driver_code}: {e}")
        return None


# ── Standings ─────────────────────────────────────────────────────────────────

def get_standings(year, up_to_round=None):
    """
    Berechnet Fahrer- und Konstrukteurs-WM aus den geladenen Sessions.
    """
    POINTS_MAP = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    sched = get_schedule(year)
    
    drv_pts   = {}
    drv_info  = {}
    team_pts  = {}

    for race in sched:
        if not race['completed']:
            continue
        if up_to_round and race['round'] > up_to_round:
            continue
        
        session = load_session(year, race['round'], 'R', telemetry=False)
        if session is None or len(session.results) == 0:
            continue

        for _, row in session.results.iterrows():
            code = str(row.get('Abbreviation', ''))
            pos  = int(row.get('Position', 99)) if not pd.isna(row.get('Position')) else 99
            pts  = POINTS_MAP.get(pos, 0)
            team = str(row.get('TeamName', ''))

            drv_pts[code]  = drv_pts.get(code, 0) + pts
            team_pts[team] = team_pts.get(team, 0) + pts
            drv_info[code] = {
                'name':  f"{row.get('FirstName','')} {row.get('LastName','')}".strip(),
                'team':  team,
                'color': get_driver_color(code),
            }

    drivers = sorted(
        [{'code': k, **drv_info.get(k, {'name':k,'team':'','color':'#888'}),
          'points': v} for k, v in drv_pts.items()],
        key=lambda x: -x['points']
    )
    constructors = sorted(
        [{'team': k, 'points': v} for k, v in team_pts.items()],
        key=lambda x: -x['points']
    )
    return {'drivers': drivers, 'constructors': constructors}


# ── Wetter ────────────────────────────────────────────────────────────────────

def get_weather(session):
    """Gibt weather-DataFrame zurück (optional)."""
    if session is None: return None
    try:
        w = session.weather_data
        return w if len(w) > 0 else None
    except Exception:
        return None


# ── Vollständige Race-Data für einen Tab ──────────────────────────────────────

def build_race_data(year, round_num):
    """
    Baut ein race-dict auf das identisch mit dem alten JSON-Format ist —
    Charts müssen nicht geändert werden.
    """
    session = load_session(year, round_num, 'R', telemetry=True)
    sched   = get_schedule(year)
    race_info = next((r for r in sched if r['round'] == round_num), {})

    results  = get_race_results(session)
    laps_all = get_lap_times(session)

    # Telemetrie nur für die ersten 6 Fahrer (Performance)
    top6 = [r['code'] for r in results[:6]]
    telemetry = {}
    for code in top6:
        t = get_telemetry(session, code)
        if t: telemetry[code] = t

    return {
        **race_info,
        'results':   results,
        'lap_times': laps_all,
        'telemetry': telemetry,
        'session':   session,    # Rohzugang für Erweiterungen
    }


if __name__ == '__main__':
    # Schnelltest
    print("Teste FastF1 Schedule 2026...")
    sched = get_schedule(2026)
    print(f"  {len(sched)} Rennen gefunden")
    for r in sched[:3]:
        print(f"  R{r['round']} {r['name']} — {r['date']} {'[OK]' if r['completed'] else '[upcoming]'}")

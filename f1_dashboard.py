"""
F1 Race Engineer Dashboard 2026 — v3
Responsiv · Dunkles Theme · Telemetrie-Tabs · FastF1 Cache
"""
import json, os, warnings, logging, hashlib, sys, pickle
import joblib
from pathlib import Path
# Windows terminals default to cp1252 and cannot render some unicode symbols;
# force UTF-8 so log output is identical on Linux and Windows.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
warnings.filterwarnings('ignore')

# ── ML models (loaded once at startup; dashboard degrades gracefully if absent) ─
_ML_READY = False
_rf_model = None
_xgb_model = None
_MODEL_META = {}

ML_MISSING_BANNER = "[WARNING] Run: python ml_models.py --demo  (saves to models/ folder)"

def _load_ml_models():
    """Returns True if models are loaded, False otherwise."""
    return _ML_READY

# ── Load models immediately at module level (not inside __main__) ─────────────
def _init_models():
    global _ML_READY, _rf_model, _xgb_model, _MODEL_META
    _dir = Path(__file__).resolve().parent
    _mdir = _dir / "models"
    rf_path  = _mdir / "random_forest_model.pkl"
    xgb_path = _mdir / "xgboost_model.pkl"
    print(f"[ML] Looking for models in: {_mdir}")
    print(f"[ML] RF exists: {rf_path.exists()}, XGB exists: {xgb_path.exists()}")
    if rf_path.exists() and xgb_path.exists():
        try:
            _rf_model  = joblib.load(rf_path)
            _xgb_model = joblib.load(xgb_path)
            meta_path = _mdir / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    _MODEL_META = json.load(f)
            _ML_READY = True
            print(f"[ML] Models loaded OK (RF={type(_rf_model).__name__}, XGB={type(_xgb_model).__name__})")
        except Exception as e:
            print(f"[ML] ERROR loading models: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"[ML] Model files not found — run: python ml_models.py --demo")

_init_models()

# ── Feature matrix + news context (lazy, shared across callbacks) ──────────────
_BASE_DF = None
_CONTEXT = None
_SIM_SUMMARY = None

def _get_base_df():
    """Historical 18-feature matrix from the demo data (built once)."""
    global _BASE_DF
    if _BASE_DF is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ml_models import build_feature_matrix, load_demo_data
        _BASE_DF = build_feature_matrix(load_demo_data())
    return _BASE_DF

def _get_context():
    """News-context updates (context_updates.json); {} if absent."""
    global _CONTEXT
    if _CONTEXT is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'context_updates.json')
        _CONTEXT = {}
        if os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    _CONTEXT = json.load(f)
            except Exception:
                _CONTEXT = {}
        _CONTEXT.setdefault('team_updates', [])
        _CONTEXT.setdefault('driver_updates', [])
    return _CONTEXT

def _get_sim_summary():
    """Season-level win probabilities from championship_prediction.json, if present."""
    global _SIM_SUMMARY
    if _SIM_SUMMARY is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'championship_prediction.json')
        _SIM_SUMMARY = {}
        if os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    _SIM_SUMMARY = json.load(f)
            except Exception:
                _SIM_SUMMARY = {}
    return _SIM_SUMMARY

import fastf1
CACHE_DIR = os.path.join(os.getcwd(), 'f1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.set_log_level('WARNING')

# ── Farben ────────────────────────────────────────────────────────────────────
C = {
    # Backgrounds
    "bg":        "#0e0e0e",
    "card":      "rgba(255,255,255,0.03)",
    "card_hover":"rgba(255,255,255,0.06)",
    "sidebar":   "rgba(14,14,14,0.85)",

    # Borders
    "border":    "rgba(255,255,255,0.08)",
    "border2":   "rgba(255,255,255,0.04)",

    # Primary palette
    "coral":     "#ffb4a7",
    "cyan":      "#7ef4f4",
    "white":     "#f0f0f0",
    "muted":     "#5a5a6a",
    "gold":      "#ffd080",

    # Keep for team colors (used elsewhere, do not remove)
    "accent":    "#E8002D",
    "orange":    "#FF8000",
    "text":      "#f0f0f0",
    "text2":     "#8888a8",

    # Legacy keys used by other tabs — keep compatible
    "panel":     "#141414",
    "grid":      "rgba(255,255,255,0.05)",
    "input_bg":  "#1a1a1a",
    "input_fg":  "#f0f0f0",
}

DRIVER_COLORS = {
    'NOR':'#FF8000','PIA':'#FF8000','VER':'#3671C6','LAW':'#3671C6',
    'LEC':'#E8002D','SAI':'#E8002D','HAM':'#E8002D',
    'RUS':'#27F4D2','ANT':'#27F4D2','BOT':'#27F4D2',
    'ALO':'#358C75','STR':'#358C75','GAS':'#0093CC','DOO':'#0093CC',
    'ALB':'#64C4FF','COL':'#64C4FF','HUL':'#52E252','BOR':'#52E252',
    'TSU':'#6692FF','HAD':'#6692FF','RIC':'#6692FF',
    'MAG':'#B6BABD','BEA':'#B6BABD','OCO':'#0093CC',
}

# Pill-specific team colours (higher contrast for dark-background pills)
PILL_COLORS = {
    'ANT': '#00D2BE', 'RUS': '#00D2BE',   # Mercedes
    'HAM': '#DC0000', 'LEC': '#DC0000',   # Ferrari
    'NOR': '#FF8000', 'PIA': '#FF8000',   # McLaren
    'VER': '#3671C6', 'HAD': '#3671C6',   # Red Bull
    'ALO': '#006F62', 'STR': '#006F62',   # Aston Martin
    'GAS': '#0093CC', 'COL': '#0093CC',   # Alpine
    'ALB': '#37BEDD', 'SAI': '#37BEDD',   # Williams
    'LAW': '#6692FF', 'LIN': '#6692FF',   # Racing Bulls
    'BEA': '#B6BABD', 'OCO': '#B6BABD',   # Haas
    'BOR': '#52E252', 'HUL': '#52E252',   # Audi
    'BOT': '#4CAF7D', 'PER': '#4CAF7D',   # Cadillac
    'DOO': '#0093CC', 'ZHO': '#52E252',   # legacy aliases
}
TEAM_COLORS = {
    'McLaren':'#FF8000','Ferrari':'#E8002D','Red Bull Racing':'#3671C6',
    'Red Bull':'#3671C6','Mercedes':'#27F4D2','Aston Martin':'#358C75',
    'Alpine':'#0093CC','Williams':'#64C4FF','Kick Sauber':'#52E252',
    'Sauber':'#52E252','RB':'#6692FF','Racing Bulls':'#6692FF',
    'Haas F1 Team':'#B6BABD','Haas':'#B6BABD',
}


def hex_alpha(h, a=0.2):
    h = h.lstrip('#')
    if len(h) == 6:
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"
    return '#' + h

def dc(code): return DRIVER_COLORS.get(code, '#888888')
def tc(team): return TEAM_COLORS.get(team, '#888888')

# ── Race data: real file if present, demo fallback otherwise ──────────────────
def _load_dashboard_data() -> dict:
    _dir = os.path.dirname(os.path.abspath(__file__))
    real = os.path.join(_dir, "f1_data_2026.json")
    demo = os.path.join(_dir, "f1_data_2026_demo.json")
    if os.path.exists(real):
        with open(real, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] Using real race data ({len(data['races'])} rounds)")
        return data
    else:
        with open(demo, encoding="utf-8") as f:
            data = json.load(f)
        print("[INFO] Using demo race data — run scripts/fetch_2026_results.py for real data")
        return data

DEMO = _load_dashboard_data()
DEMO_BY_ROUND = {r['round']: r for r in DEMO['races']}

ALL_DRIVER_CODES = [d['code'] for d in DEMO['drivers']]

# ── FastF1 Session Cache ──────────────────────────────────────────────────────
_sess_cache = {}

def load_session(year, round_num, stype='R'):
    key = (year, round_num, stype)
    if key in _sess_cache:
        return _sess_cache[key]
    try:
        s = fastf1.get_session(year, round_num, stype)
        s.load(laps=True, telemetry=True, weather=True, messages=False)
        if len(s.results) > 0 or len(s.laps) > 0:
            _sess_cache[key] = s
            return s
    except Exception:
        pass
    return None

def _td(td):
    if pd.isna(td): return None
    return td.total_seconds() if hasattr(td, 'total_seconds') else float(td)

def get_race_data(year, round_num):
    """Echte FastF1-Daten oder Demo-Fallback"""
    sess = load_session(year, round_num, 'R')
    demo = DEMO_BY_ROUND.get(round_num, DEMO['races'][-1])
    if sess is None or len(sess.results) == 0:
        return demo

    PTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    results, laps_all, telemetry = [], {}, {}
    for _, row in sess.results.iterrows():
        code = str(row.get('Abbreviation',''))
        pos  = int(row.get('Position',99)) if not pd.isna(row.get('Position',np.nan)) else 99
        gap  = 0.0
        if pos > 1:
            gap = _td(row.get('Time') or row.get('GapToLeader')) or 0.0
        dl   = sess.laps.pick_driver(code) if len(sess.laps) > 0 else pd.DataFrame()
        fl_t, secs = None, {'S1':30.,'S2':40.,'S3':28.}
        if len(dl) > 0:
            try:
                fl = dl.pick_fastest()
                fl_t = _td(fl['LapTime'])
                secs = {
                    'S1': _td(fl['Sector1Time']) or 30.,
                    'S2': _td(fl['Sector2Time']) or 40.,
                    'S3': _td(fl['Sector3Time']) or 28.,
                }
            except Exception: pass
        team = str(row.get('TeamName',''))
        results.append({
            'position': pos, 'code': code,
            'name': f"{row.get('FirstName','')} {row.get('LastName','')}".strip(),
            'team': team, 'color': dc(code) or tc(team),
            'points': PTS.get(pos,0), 'gap': round(gap,3),
            'fastest_lap': fl_t or 90., 'sectors': secs,
            'pit_stops': max(1,int(dl['PitOutTime'].notna().sum())) if len(dl)>0 else 1,
            'tyre_strategy': _tyre_strat(dl),
        })
    results.sort(key=lambda x: x['position'])

    for code in sess.laps['Driver'].unique():
        dl = sess.laps.pick_driver(code)
        times = [round(_td(t),3) for t in dl['LapTime'] if _td(t) and 60 < _td(t) < 200]
        if times: laps_all[code] = times

    for r in results[:6]:
        code = r['code']
        try:
            fl  = sess.laps.pick_driver(code).pick_fastest()
            tel = fl.get_telemetry().add_distance()
            telemetry[code] = {
                'distance': tel['Distance'].round(1).tolist(),
                'speed':    tel['Speed'].tolist(),
                'throttle': tel['Throttle'].tolist(),
                'brake':    (tel['Brake'].astype(float)*100).tolist(),
                'gear':     tel['nGear'].tolist(),
                'drs':      tel['DRS'].tolist(),
                'rpm':      tel['RPM'].tolist() if 'RPM' in tel else [],
            }
        except Exception: pass

    return {**demo, 'results': results, 'lap_times': laps_all, 'telemetry': telemetry}

def _tyre_strat(dl):
    if len(dl) == 0: return 'M-H'
    try:
        seen, out = set(), []
        for c in dl['Compound'].dropna():
            a = c[0]
            if a not in seen: seen.add(a); out.append(a)
        return '-'.join(out) or 'M-H'
    except: return 'M-H'

# ── Standings: Jolpica API ────────────────────────────────────────────────────
def get_live_standings(year=2026):
    import urllib.request, json as _json
    try:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
        req = urllib.request.Request(url, headers={'User-Agent': 'F1Dashboard/2.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = _json.loads(r.read())
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        drivers = []
        for s in standings:
            code = s['Driver'].get('code','???')
            drivers.append({
                'code': code,
                'name': f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                'team': s['Constructors'][0]['name'] if s['Constructors'] else '',
                'color': dc(code),
                'points': int(float(s['points'])),
                'wins': int(s['wins']),
            })
        return {'drivers': drivers}
    except Exception:
        return None

def get_live_constructor_standings(year=2026):
    import urllib.request, json as _json
    try:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json"
        req = urllib.request.Request(url, headers={'User-Agent': 'F1Dashboard/2.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = _json.loads(r.read())
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        return [{'team': s['Constructor']['name'], 'points': int(float(s['points']))} for s in standings]
    except Exception:
        return None

# ── Schedule ──────────────────────────────────────────────────────────────────
_sched_cache = {}
def get_schedule(year):
    if year in _sched_cache: return _sched_cache[year]
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        today = pd.Timestamp.now().tz_localize(None)
        races = []
        for _, ev in sched.iterrows():
            d = pd.Timestamp(ev['EventDate'])
            if d.tzinfo: d = d.tz_localize(None)
            races.append({
                'round': int(ev['RoundNumber']), 'name': ev['EventName'],
                'circuit': ev.get('CircuitShortName', ev['EventName'].replace(' Grand Prix','')),
                'country': ev['Country'], 'date': d.strftime('%Y-%m-%d'),
                'completed': d < today,
            })
        _sched_cache[year] = sorted(races, key=lambda x: x['round'])
        return _sched_cache[year]
    except Exception:
        return [{'round':r['round'],'name':r['name'],'circuit':r.get('circuit',''),'country':r.get('country',''),
                 'date':r.get('date',''),'completed':True} for r in DEMO['races']]

# ── Plot-Basis Layout ─────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['card'],
    font=dict(color=C['text'], family='Inter, system-ui, sans-serif', size=11),
    xaxis=dict(gridcolor=C['grid'], linecolor=C['border2'], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C['grid'], linecolor=C['border2'], showgrid=True, zeroline=False),
    margin=dict(l=48,r=16,t=36,b=40),
    legend=dict(bgcolor='rgba(19,19,31,0.9)', bordercolor=C['border2'], borderwidth=1, font=dict(size=10)),
    hoverlabel=dict(bgcolor=C['panel'], font_size=11, font_color=C['text'], bordercolor=C['accent']),
)

def pl(**kw): return {**PL, **kw}

# ── UI Helpers ────────────────────────────────────────────────────────────────
def card(*children, style=None):
    base = {
        'background': C['card'],
        'border': f'1px solid {C["border"]}',
        'borderRadius': '16px',
        'backdropFilter': 'blur(20px)',
        'WebkitBackdropFilter': 'blur(20px)',
        'padding': '20px 24px',
    }
    return html.Div(children, style={**base, **(style or {})})

def shead(title, icon=''):
    return html.Div([
        html.Span(icon, style={'marginRight':'8px','fontSize':'12px'}),
        html.Span(title, style={
            'fontFamily':"'Barlow Condensed',sans-serif",
            'fontWeight':'700','fontSize':'10px','letterSpacing':'2.5px',
            'color':'rgba(255,255,255,0.3)','textTransform':'uppercase',
        }),
    ], style={'marginBottom':'14px'})

def kpi(label, value, color=None):
    return html.Div([
        html.Div(label, style={'fontSize':'9px','color':C['muted'],'letterSpacing':'1.5px',
                               'textTransform':'uppercase','marginBottom':'4px'}),
        html.Div(str(value), style={'fontSize':'20px','fontWeight':'700',
                                    'color': color or C['orange'],'lineHeight':'1.1'}),
    ], style={'background':C['card'],'border':f'1px solid {C["border"]}',
              'borderTop':f'2px solid {color or C["orange"]}',
              'borderRadius':'6px','padding':'12px 16px','flex':'1','minWidth':'110px'})

def lap_str(s):
    m,sec = divmod(abs(float(s)),60)
    return f"{int(m)}:{sec:06.3f}"

# ── Fahrer-Pill-Selektor ──────────────────────────────────────────────────────
def driver_pills(d1, d2):
    """Zwei Spalten Pill-Buttons nebeneinander; d1/d2 aktiv hervorgehoben."""
    def pill(code, which, active_code):
        col = PILL_COLORS.get(code, dc(code))
        is_active = code == active_code
        return html.Button(
            code,
            id={"type": f"d{which}-pill", "code": code},
            n_clicks=0,
            className='driver-pill',
            style={
                '--pill-hover-bg': hex_alpha(col, 0.2),
                'border': f'2px solid {col}',
                'background': col if is_active else 'transparent',
                'color': '#0D0D0D' if is_active else col,
                'borderRadius': '20px',
                'padding': '7px 14px',
                'fontSize': '13px',
                'fontWeight': '700',
                'cursor': 'pointer',
                'fontFamily': 'monospace',
                'letterSpacing': '0.5px',
                'lineHeight': '1.4',
                'transition': 'all 0.12s',
                'whiteSpace': 'nowrap',
                'minWidth': '58px',
            }
        )

    def column(which, active_code):
        label = 'Q1' if which == 1 else 'Q2'
        return html.Div([
            html.Div(label, style={
                'fontSize': '11px',
                'color': '#FFFFFF',
                'fontWeight': '700',
                'letterSpacing': '2px',
                'textTransform': 'uppercase',
                'marginBottom': '6px',
            }),
            html.Div(
                [pill(code, which, active_code) for code in ALL_DRIVER_CODES],
                className='pill-row',
                style={
                    'display': 'flex', 'flexWrap': 'nowrap',
                    'gap': '5px', 'overflowX': 'auto',
                    'paddingBottom': '4px',
                },
            ),
        ], style={'flex': '1', 'minWidth': '0'})

    return html.Div([
        column(1, d1),
        column(2, d2),
    ], style={
        'display': 'flex',
        'gap': '16px',
        'background': C['panel'],
        'border': f'1px solid {C["border"]}',
        'borderRadius': '8px',
        'padding': '12px 16px',
        'marginBottom': '14px',
        'flexWrap': 'wrap',
    })


# ── Telemetrie-Kanal-Tabs ─────────────────────────────────────────────────────
TEL_CHANNELS = {
    'speed':    ('Speed',    'km/h'),
    'throttle': ('Throttle', '%'),
    'brake':    ('Brake',    '%'),
    'gear':     ('Gear',     ''),
    'drs':      ('DRS',      ''),
    'rpm':      ('RPM',      'RPM'),
}

def tel_tab_bar(active_channel):
    return html.Div([
        html.Button(
            label,
            id={"type": "tel-tab", "channel": ch},
            n_clicks=0,
            style={
                'background': 'rgba(232,0,45,0.12)' if ch == active_channel else 'transparent',
                'border': f'1px solid {C["accent"]}' if ch == active_channel else f'1px solid {C["border2"]}',
                'borderRadius': '5px',
                'color': C['accent'] if ch == active_channel else C['muted'],
                'padding': '6px 18px',
                'fontSize': '12px',
                'fontWeight': '600' if ch == active_channel else '400',
                'cursor': 'pointer',
                'letterSpacing': '0.5px',
                'transition': 'all 0.1s',
                'minWidth': '80px',
            }
        )
        for ch, (label, _) in TEL_CHANNELS.items()
    ], style={'display':'flex','gap':'6px','flexWrap':'wrap','marginBottom':'16px'})


# ── CSS ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(os.getcwd(),'assets'), exist_ok=True)
with open(os.path.join(os.getcwd(),'assets','v2.css'),'w', encoding='utf-8') as f:
    f.write(f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{C['bg']};color:{C['text']};
  font-family:'Barlow Condensed',sans-serif;overflow-x:hidden}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:{C['bg']}}}
::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.15);border-radius:3px}}

/* Sidebar */
#sidebar{{width:200px;min-width:200px;height:100vh;
  background:{C['sidebar']};backdrop-filter:blur(20px);
  border-right:1px solid {C['border']};display:flex;flex-direction:column;
  position:fixed;top:0;left:0;z-index:200;overflow-y:auto}}
#content{{margin-left:200px;min-height:100vh;background:{C['bg']}}}

/* Nav items */
.nav-item{{display:flex;align-items:center;gap:10px;padding:9px 18px;
  cursor:pointer;color:rgba(255,255,255,0.35);font-size:12px;font-weight:500;
  border-left:2px solid transparent;transition:all 0.15s;
  font-family:'Barlow Condensed',sans-serif;letter-spacing:0.5px}}
.nav-item:hover{{color:{C['text']};background:rgba(255,255,255,0.04)}}
.nav-active{{color:{C['coral']}!important;background:rgba(255,180,167,0.08)!important;
  border-left:2px solid {C['coral']}!important}}

/* ── Dropdowns — Dash 4 kompatibel ───────────────────────────────────────── */
.dash-dropdown,
.dash-dropdown .Select,
.dash-dropdown .Select-control {{
  background:{C['input_bg']}!important;
  border-color:{C['border']}!important;
  color:{C['input_fg']}!important;
  font-family:'Barlow Condensed',sans-serif!important;
}}
.dash-dropdown .Select-control {{
  border:1px solid {C['border']}!important;
  border-radius:8px!important;
}}
[class*="menu"] {{
  background:#1a1a1a!important;
  border:1px solid {C['border']}!important;
  z-index:9999!important;
}}
[class*="option"] {{
  background:#1a1a1a!important;
  color:{C['input_fg']}!important;
  font-family:'Barlow Condensed',sans-serif!important;
}}
[class*="option"]:hover,
[class*="option--is-focused"] {{
  background:rgba(255,255,255,0.06)!important;
  color:{C['text']}!important;
}}
[class*="option--is-selected"] {{
  background:rgba(255,180,167,0.15)!important;
  color:{C['coral']}!important;
}}
[class*="single-value"] {{
  color:{C['input_fg']}!important;
  font-family:'Barlow Condensed',sans-serif!important;
}}
[class*="placeholder"] {{
  color:{C['muted']}!important;
}}
.dash-dropdown input[type="text"],
.dash-dropdown input {{
  color:{C['input_fg']}!important;
  background:transparent!important;
}}
.dash-dropdown .Select-arrow-zone {{color:{C['muted']}!important}}
.dash-dropdown .Select--multi .Select-value {{
  background:rgba(255,180,167,0.15)!important;
  border-color:rgba(255,180,167,0.3)!important;
  color:{C['coral']}!important;
}}
.dash-dropdown .Select-clear {{color:{C['muted']}!important}}

/* Topbar */
#topbar{{display:flex;align-items:center;justify-content:space-between;
  padding:12px 22px;background:rgba(14,14,14,0.9);backdrop-filter:blur(20px);
  border-bottom:1px solid {C['border']};position:sticky;top:0;z-index:100}}

/* Responsive grid */
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}}
@media(max-width:1100px){{.grid-2{{grid-template-columns:1fr}}.grid-3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:750px){{.grid-3{{grid-template-columns:1fr}}#sidebar{{width:160px}}#content{{margin-left:160px}}}}

/* Tabs */
.tab-btn{{background:transparent;border:1px solid {C['border']};border-radius:8px;
  color:{C['muted']};padding:5px 12px;font-size:11px;cursor:pointer;
  margin-right:6px;margin-bottom:6px;font-family:'Barlow Condensed',sans-serif;
  letter-spacing:0.5px}}
.tab-btn:hover{{border-color:rgba(255,255,255,0.2);color:{C['text']}}}
.tab-btn.active{{background:rgba(255,180,167,0.1);border-color:{C['coral']};color:{C['coral']}}}

/* Table */
.f1-table{{width:100%;border-collapse:collapse;font-size:12px}}
.f1-table th{{color:rgba(255,255,255,0.3);font-weight:700;font-size:9px;letter-spacing:2px;
  text-transform:uppercase;padding:8px 10px;
  border-bottom:1px solid {C['border']};text-align:left;
  font-family:'Barlow Condensed',sans-serif}}
.f1-table td{{padding:7px 10px;border-bottom:1px solid {C['border']};color:{C['text']}}}
.f1-table tr:hover td{{background:rgba(255,255,255,0.02)}}

/* Driver pills */
.driver-pill:hover{{background:var(--pill-hover-bg) !important;}}
/* Thin scrollbar for pill rows */
.pill-row::-webkit-scrollbar{{height:3px}}
.pill-row::-webkit-scrollbar-track{{background:transparent}}
.pill-row::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.15);border-radius:2px}}

/* Animations */
@keyframes fadeIn{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
.fadein{{animation:fadeIn 0.2s ease}}

/* Label */
.sidebar-label{{font-size:9px;color:rgba(255,255,255,0.25);letter-spacing:2px;
  text-transform:uppercase;padding:14px 18px 5px;
  font-family:'Barlow Condensed',sans-serif}}

/* Glass card */
.glass-card{{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
  border-radius:16px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  padding:20px 24px}}

/* Mono nums */
.mono{{font-family:'JetBrains Mono',monospace}}
""")

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__,
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets'),
    external_stylesheets=[dbc.themes.CYBORG,
        'https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap'],
    title='F1 Race Engineer 2026',
    suppress_callback_exceptions=True,
)

# Nav config
NAV = [
    ('overview',    '▦', 'Race Overview'),
    ('laps',        '▲', 'Lap Analysis'),
    ('telemetry',   '≋', 'Telemetry'),
    ('sectors',     '◔', 'Sector Times'),
    ('car',         '◈', 'Car Performance'),
    ('energy',      '↯', 'Energy / ERS'),
    ('strategy',    '◍', 'Strategy'),
    ('track',       '◌', 'Track Dominance'),
    ('standings',   '★', 'Standings'),
    ('predictions', '◆', 'ML Predictions'),
    ('live',        '●', 'Live Monitor'),
]

def sidebar():
    return html.Div([
        html.Div([
            html.Div('F1', style={'fontSize':'18px','fontWeight':'900',
                                  'color':C['accent'],'letterSpacing':'1px'}),
            html.Div([
                html.Div('RACE ENGINEER', style={'fontSize':'9px','color':C['accent'],'fontWeight':'700','letterSpacing':'2.5px'}),
                html.Div('2026 Season', style={'fontSize':'15px','fontWeight':'700'}),
            ]),
        ], style={'display':'flex','gap':'10px','alignItems':'center','padding':'16px 18px','borderBottom':f'1px solid {C["border"]}'}),

        html.Div('Navigation', className='sidebar-label'),
        *[html.Div([
            html.Span(icon, style={'fontSize':'13px','width':'18px'}),
            html.Span(label, style={'fontSize':'12px'}),
        ], id=f'nav-{tid}', n_clicks=0, className='nav-item')
        for tid, icon, label in NAV],

        html.Div('Saison', className='sidebar-label'),
        dcc.Dropdown(id='sel-year',
            options=[{'label':str(y),'value':y} for y in [2026,2025,2024,2023]],
            value=2026, clearable=False, className='dark-dropdown',
            style={'margin':'0 10px 8px','fontSize':'12px'}),

        html.Div('Grand Prix', className='sidebar-label'),
        dcc.Dropdown(id='sel-race', clearable=False, className='dark-dropdown',
            style={'margin':'0 10px 8px','fontSize':'12px'}),

        html.Div(style={'flex':'1'}),
        html.Div('FastF1 v3.8.3', style={'fontSize':'9px','color':C['muted'],
            'padding':'10px 18px','borderTop':f'1px solid {C["border"]}'}),
    ], id='sidebar')

app.layout = html.Div([
    dcc.Store(id='store-tab',        data='overview'),
    dcc.Store(id='store-race-data',  data={}),
    dcc.Store(id='sel-d1',           data='NOR'),
    dcc.Store(id='sel-d2',           data='LEC'),
    dcc.Store(id='tel-channel',      data='speed'),
    sidebar(),
    html.Div([
        html.Div([
            html.Div(id='topbar-title', style={'fontWeight':'600','fontSize':'14px'}),
            html.Div(id='topbar-meta', style={'fontSize':'11px','color':C['muted']}),
        ], id='topbar'),
        html.Div(id='page-content', style={'padding':'18px 22px'}),
    ], id='content'),
])

# ── Race-Dropdown updaten wenn Jahr wechselt ──────────────────────────────────
@app.callback(
    [Output('sel-race','options'), Output('sel-race','value')],
    Input('sel-year','value')
)
def update_race_options(year):
    sched = get_schedule(year)
    race_opts = [{'label':f"R{r['round']} · {r['name'].replace(' Grand Prix',' GP')}", 'value':r['round']} for r in sched]
    completed = [r['round'] for r in sched if r.get('completed')]
    default_race = completed[-1] if completed else (sched[-1]['round'] if sched else 1)
    return race_opts, default_race

# ── Fahrer 1 Pill-Buttons ─────────────────────────────────────────────────────
@app.callback(
    Output('sel-d1', 'data'),
    Input({"type": "d1-pill", "code": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_d1(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered: return no_update
    triggered = ctx.triggered[0]
    if not triggered['value']:   # 0 oder None → neu erstellte Komponente, kein echter Klick
        return no_update
    prop = triggered['prop_id']
    try:
        code = json.loads(prop.split('.')[0])['code']
        return code
    except Exception:
        return no_update

# ── Fahrer 2 Pill-Buttons ─────────────────────────────────────────────────────
@app.callback(
    Output('sel-d2', 'data'),
    Input({"type": "d2-pill", "code": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_d2(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered: return no_update
    triggered = ctx.triggered[0]
    if not triggered['value']:   # 0 oder None → neu erstellte Komponente, kein echter Klick
        return no_update
    prop = triggered['prop_id']
    try:
        code = json.loads(prop.split('.')[0])['code']
        return code
    except Exception:
        return no_update

# ── Telemetrie-Kanal-Tab ──────────────────────────────────────────────────────
@app.callback(
    Output('tel-channel', 'data'),
    Input({"type": "tel-tab", "channel": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_tel_channel(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered: return no_update
    triggered = ctx.triggered[0]
    if not triggered['value']:
        return no_update
    prop = triggered['prop_id']
    try:
        ch = json.loads(prop.split('.')[0])['channel']
        return ch
    except Exception:
        return no_update

# ── Nav: aktiver Tab ──────────────────────────────────────────────────────────
@app.callback(
    Output('store-tab','data'),
    [Input(f'nav-{t}','n_clicks') for t,_,_ in NAV],
    prevent_initial_call=True,
)
def set_tab(*args):
    ctx = callback_context
    if not ctx.triggered: return no_update
    tid = ctx.triggered[0]['prop_id'].split('.')[0].replace('nav-','')
    return tid

@app.callback(
    [Output(f'nav-{t}','className') for t,_,_ in NAV],
    Input('store-tab','data'),
)
def update_nav_classes(tab):
    return ['nav-item nav-active' if t==tab else 'nav-item' for t,_,_ in NAV]

# ── Race-Daten laden ──────────────────────────────────────────────────────────
@app.callback(
    [Output('store-race-data','data'), Output('topbar-meta','children')],
    [Input('sel-race','value'), Input('sel-year','value')],
)
def load_race(round_num, year):
    if round_num is None: return {}, ''
    data = get_race_data(year, round_num)
    meta = f"R{data.get('round','?')} · {data.get('circuit','?')} · {data.get('date','?')[:7]}"
    light = {k:v for k,v in data.items() if k not in ('telemetry',)}
    return light, meta

# ── Hauptinhalt rendern ───────────────────────────────────────────────────────
@app.callback(
    [Output('page-content','children'), Output('topbar-title','children')],
    [Input('store-tab','data'), Input('store-race-data','data'),
     Input('sel-d1','data'), Input('sel-d2','data'),
     Input('sel-year','value'), Input('sel-race','value'),
     Input('tel-channel','data')],
)
def render_page(tab, race_data, d1, d2, year, round_num, tel_channel):
    if not round_num: return html.Div("Rennen wählen…"), "F1 Race Engineer"
    if not race_data:
        race_data = get_race_data(year or 2026, round_num)

    d1 = d1 or 'NOR'
    d2 = d2 or 'LEC'
    tab = tab or 'overview'
    tel_channel = tel_channel or 'speed'

    title_map = {t:label for t,_,label in NAV}
    page_title = f"{title_map.get(tab,'?')} — {race_data.get('name','?')}"

    builders = {
        'overview':    build_overview,
        'laps':        build_laps,
        'telemetry':   build_telemetry,
        'sectors':     build_sectors,
        'car':         build_car,
        'energy':      build_energy,
        'strategy':    build_strategy,
        'track':       build_track,
        'standings':   build_standings,
        'predictions': build_predictions,
        'live':        build_live,
    }
    try:
        content = builders.get(tab, build_overview)(
            race_data, d1, d2, year, round_num, tel_channel=tel_channel
        )
    except Exception as e:
        content = html.Div(f"Fehler: {e}", style={'color':C['accent'],'padding':'20px'})

    return html.Div(content, className='fadein'), page_title

# ══════════════════════════════════════════════════════════════════════════════
# ── TAB BUILDERS ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_overview(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    lap_times = rd.get('lap_times',{})
    if not results: return html.Div("Keine Daten", style={'color':C['muted'],'padding':'30px'})

    df = pd.DataFrame(results)
    winner = results[0]
    r1 = next((r for r in results if r['code']==d1), results[0])
    r2 = next((r for r in results if r['code']==d2), results[1] if len(results)>1 else results[0])
    fastest = min(results, key=lambda r: r['fastest_lap'])

    fig_gap = go.Figure(go.Bar(
        x=df['code'], y=df['gap'],
        marker_color=[dc(c) for c in df['code']],
        text=[f"+{v:.2f}" if v>0 else 'P1' for v in df['gap']],
        textposition='outside', textfont=dict(size=9),
        hovertemplate='<b>%{x}</b>  +%{y:.3f}s<extra></extra>',
    ))
    fig_gap.update_layout(**pl(height=240,title=dict(text='Gap to Winner',font=dict(size=10,color=C['muted'])),
        yaxis_title='Gap (s)', xaxis_tickfont=dict(size=9)))

    fig_box = go.Figure()
    for r in results[:8]:
        times = [t for t in lap_times.get(r['code'],[]) if t<97]
        if times:
            col = dc(r['code'])
            fig_box.add_trace(go.Box(
                y=times, name=r['code'],
                marker_color=col, line_color=col,
                fillcolor=hex_alpha(col,0.13), boxpoints='outliers',
                hovertemplate='<b>%{x}</b><br>%{y:.3f}s<extra></extra>',
            ))
    fig_box.update_layout(**pl(height=280, showlegend=False,
        title=dict(text='Race Pace — Lap Time Distribution',font=dict(size=10,color=C['muted'])),
        yaxis_title='Lap Time (s)'))

    rows = []
    for r in results[:10]:
        rows.append(html.Tr([
            html.Td(f"P{r['position']}", style={'color':C['accent'] if r['position']==1 else C['muted'],'fontWeight':'700'}),
            html.Td(html.Span(r['code'], style={'color':dc(r['code']),'fontWeight':'600'})),
            html.Td(r['team'], style={'color':C['muted'],'fontSize':'11px'}),
            html.Td(f"+{r['gap']:.3f}s" if r['gap']>0 else '—', style={'fontFamily':'monospace','fontSize':'11px'}),
            html.Td(lap_str(r['fastest_lap']), style={'fontFamily':'monospace','fontSize':'11px','color':C['muted']}),
            html.Td(str(r['points']), style={'color':C['orange'] if r['points']>0 else C['muted'],'fontWeight':'600'}),
            html.Td(r['tyre_strategy'], style={'fontSize':'10px','color':C['muted']}),
        ]))

    return [
        driver_pills(d1, d2),
        html.Div([
            kpi('Winner', winner['code'], C['accent']),
            kpi('Fastest Lap', fastest['code'], '#ffdd00'),
            kpi(f'{d1} Pos', f"P{r1['position']}", dc(d1)),
            kpi(f'{d2} Pos', f"P{r2['position']}", dc(d2)),
            kpi(f'{d1} Pts', r1['points'], dc(d1)),
            kpi(f'{d2} Pts', r2['points'], dc(d2)),
        ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'16px'}),

        html.Div([
            card(shead('Race Results','▸'),
                html.Table([
                    html.Thead(html.Tr([html.Th(h) for h in ['Pos','Drv','Team','Gap','Best Lap','Pts','Strategy']])),
                    html.Tbody(rows),
                ], className='f1-table'),
            style={'flex':'1.3'}),
            card(shead('Gap to Winner','▦'), dcc.Graph(figure=fig_gap, config={'displayModeBar':False}),
                style={'flex':'1'}),
        ], className='grid-2', style={'marginBottom':'14px'}),

        card(shead('Lap Time Distribution (Box Plots)','▤'),
            dcc.Graph(figure=fig_box, config={'displayModeBar':False})),
    ]

def build_laps(rd, d1, d2, year, round_num, **kwargs):
    lap_times = rd.get('lap_times',{})
    if not lap_times: return html.Div("Keine Rundenzeitdaten verfügbar.", style={'color':C['muted'],'padding':'30px'})

    fig_pace = go.Figure()
    for code, times in lap_times.items():
        is_sel = code in (d1, d2)
        col = dc(code)
        fig_pace.add_trace(go.Scatter(
            y=times, x=list(range(1,len(times)+1)), mode='lines',
            name=code, line=dict(color=col, width=2.5 if is_sel else 1),
            opacity=1.0 if is_sel else 0.2, showlegend=is_sel,
            hovertemplate=f'<b>{code}</b> L%{{x}}: %{{y:.3f}}s<extra></extra>',
        ))
    fig_pace.update_layout(**pl(height=320,
        title=dict(text='Race Pace — alle Fahrer',font=dict(size=10,color=C['muted'])),
        xaxis_title='Lap', yaxis_title='Lap Time (s)'))

    fig_roll = go.Figure()
    for code, col in [(d1,dc(d1)),(d2,dc(d2))]:
        times = pd.Series([t for t in lap_times.get(code,[]) if t<97])
        fig_roll.add_trace(go.Scatter(y=times.tolist(), x=list(range(1,len(times)+1)),
            mode='lines', name=f'{code} raw', line=dict(color=col,width=1,dash='dot'), opacity=0.4))
        roll = times.rolling(5,center=True).mean()
        fig_roll.add_trace(go.Scatter(y=roll.tolist(), x=list(range(1,len(roll)+1)),
            mode='lines', name=f'{code} 5-lap avg', line=dict(color=col,width=2.5)))
    fig_roll.update_layout(**pl(height=260,
        title=dict(text=f'Rolling Avg (5 Laps) — {d1} vs {d2}',font=dict(size=10,color=C['muted'])),
        xaxis_title='Lap', yaxis_title='Lap Time (s)'))

    t1 = lap_times.get(d1,[])
    t2 = lap_times.get(d2,[])
    n = min(len(t1),len(t2))
    delta = [t1[i]-t2[i] for i in range(n)]
    fig_delta = go.Figure()
    fig_delta.add_hline(y=0, line_color=C['muted'], line_width=1)
    fig_delta.add_trace(go.Bar(x=list(range(1,n+1)), y=delta,
        marker_color=[dc(d1) if v>0 else dc(d2) for v in delta],
        hovertemplate='Lap %{x}  Δ %{y:+.3f}s<extra></extra>'))
    fig_delta.update_layout(**pl(height=220,
        title=dict(text=f'Lap Delta: {d1} − {d2}',font=dict(size=10,color=C['muted'])),
        xaxis_title='Lap', yaxis_title='Δ s'))

    fig_vio = go.Figure()
    for code, col in [(d1,dc(d1)),(d2,dc(d2))]:
        times = [t for t in lap_times.get(code,[]) if t<97]
        fig_vio.add_trace(go.Violin(y=times, name=code, line_color=col,
            fillcolor=hex_alpha(col,0.20), box_visible=True, meanline_visible=True, points='outliers',
            hovertemplate='<b>%{x}</b><br>%{y:.3f}s<extra></extra>'))
    fig_vio.update_layout(**pl(height=280,
        title=dict(text=f'Lap Time Violin — {d1} vs {d2}',font=dict(size=10,color=C['muted'])),
        yaxis_title='Lap Time (s)'))

    return [
        driver_pills(d1, d2),
        card(shead('Race Pace','▲'), dcc.Graph(figure=fig_pace, config={'displayModeBar':True}),
             style={'marginBottom':'14px'}),
        html.Div([
            card(shead('Rolling Average','▼'), dcc.Graph(figure=fig_roll, config={'displayModeBar':False})),
            card(shead('Violin Distribution','≋'), dcc.Graph(figure=fig_vio, config={'displayModeBar':False})),
        ], className='grid-2', style={'marginBottom':'14px'}),
        card(shead(f'Lap Delta: {d1} vs {d2}','◇'),
             dcc.Graph(figure=fig_delta, config={'displayModeBar':False})),
    ]

def build_telemetry(rd, d1, d2, year, round_num, tel_channel='speed', **kwargs):
    sess = load_session(year, round_num, 'R')
    demo_rd = DEMO_BY_ROUND.get(round_num, DEMO['races'][-1])
    tel_store = rd.get('telemetry', demo_rd.get('telemetry', {}))

    def get_tel(code):
        if sess:
            try:
                dl = sess.laps.pick_driver(code)
                fl = dl.pick_fastest()
                t  = fl.get_telemetry().add_distance()
                return {
                    'distance': t['Distance'].round(1).tolist(),
                    'speed':    t['Speed'].tolist(),
                    'throttle': t['Throttle'].tolist(),
                    'brake':    (t['Brake'].astype(float)*100).tolist(),
                    'gear':     t['nGear'].tolist(),
                    'drs':      t['DRS'].tolist(),
                    'rpm':      t['RPM'].tolist() if 'RPM' in t.columns else [],
                }
            except Exception: pass
        return tel_store.get(code)

    t1 = get_tel(d1)
    t2 = get_tel(d2)
    c1, c2 = dc(d1), dc(d2)

    if not t1 and not t2:
        return [
            driver_pills(d1, d2),
            card(html.Div("Telemetrie nicht verfügbar — Daten noch nicht im Cache.",
                style={'color':C['muted'],'padding':'20px','textAlign':'center'}))
        ]

    # KPIs
    max1 = max(t1['speed']) if t1 and t1.get('speed') else 0
    max2 = max(t2['speed']) if t2 and t2.get('speed') else 0
    avg1 = sum(t1['speed'])/len(t1['speed']) if t1 and t1.get('speed') else 0
    avg2 = sum(t2['speed'])/len(t2['speed']) if t2 and t2.get('speed') else 0

    # Aktiven Kanal bestimmen — Fallback auf Speed wenn keine RPM-Daten
    ch = tel_channel if tel_channel else 'speed'
    if ch == 'rpm' and (not t1 or not t1.get('rpm')) and (not t2 or not t2.get('rpm')):
        ch = 'speed'

    label, unit = TEL_CHANNELS[ch]
    y_title = f'{label} ({unit})' if unit else label

    # Einzelkanal-Chart
    fig = go.Figure()
    for driver, tel, color in [(d1, t1, c1), (d2, t2, c2)]:
        if tel and ch in tel and tel[ch]:
            dist = tel.get('distance', list(range(len(tel[ch]))))
            fig.add_trace(go.Scatter(
                x=dist, y=tel[ch],
                name=driver,
                line=dict(color=color, width=2.2),
                mode='lines',
                hovertemplate=f'<b>{driver}</b>  %{{x:.0f}} m  →  %{{y:.1f}} {unit}<extra></extra>',
            ))

    fig.update_layout(**pl(
        height=420,
        title=dict(text=f'{label} — Fastest Lap Telemetry: {d1} vs {d2}',
                   font=dict(size=11, color=C['muted'])),
        xaxis_title='Distance (m)',
        yaxis_title=y_title,
        xaxis=dict(gridcolor=C['grid'], linecolor=C['border2'], showgrid=True, zeroline=False,
                   title=dict(text='Distance (m)', font=dict(size=11))),
        yaxis=dict(gridcolor=C['grid'], linecolor=C['border2'], showgrid=True, zeroline=False,
                   title=dict(text=y_title, font=dict(size=11))),
        legend=dict(bgcolor='rgba(19,19,31,0.9)', bordercolor=C['border2'], borderwidth=1,
                    font=dict(size=12), x=0.01, y=0.99),
    ))

    # Speed Delta (immer verfügbar als Zusatz unter dem Hauptchart)
    spd_delta_fig = None
    if t1 and t2 and t1.get('speed') and t2.get('speed'):
        n = min(len(t1['speed']),len(t2['speed']))
        delta = [t1['speed'][i]-t2['speed'][i] for i in range(n)]
        dist  = t1['distance'][:n] if t1.get('distance') else list(range(n))
        spd_delta_fig = go.Figure()
        spd_delta_fig.add_hline(y=0, line_color=C['muted'], line_width=1)
        spd_delta_fig.add_trace(go.Bar(x=dist, y=delta,
            marker_color=[c1 if v>0 else c2 for v in delta],
            hovertemplate='%{x:.0f}m  Δ %{y:+.1f} km/h<extra></extra>'))
        spd_delta_fig.update_layout(**pl(height=200,
            title=dict(text=f'Speed Delta: {d1} − {d2}', font=dict(size=10,color=C['muted'])),
            xaxis_title='Distance (m)', yaxis_title='Δ km/h'))

    return [
        driver_pills(d1, d2),
        html.Div([
            kpi(f'{d1} Max Speed', f'{max1:.0f} km/h', c1),
            kpi(f'{d2} Max Speed', f'{max2:.0f} km/h', c2),
            kpi(f'{d1} Avg Speed', f'{avg1:.1f} km/h', c1),
            kpi(f'{d2} Avg Speed', f'{avg2:.1f} km/h', c2),
        ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'14px'}),

        card(
            tel_tab_bar(ch),
            shead(f'Telemetry — {label}','≋'),
            dcc.Graph(figure=fig, config={'displayModeBar':True}),
            style={'marginBottom':'14px'}
        ),

        (card(
            shead(f'Speed Delta by Track Position','↯'),
            dcc.Graph(figure=spd_delta_fig, config={'displayModeBar':False})
        ) if spd_delta_fig else html.Div()),
    ]

def build_sectors(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    if not results: return html.Div("Keine Daten", style={'color':C['muted'],'padding':'30px'})

    top = results[:10]
    codes = [r['code'] for r in top]
    s1 = [r['sectors']['S1'] for r in top]
    s2 = [r['sectors']['S2'] for r in top]
    s3 = [r['sectors']['S3'] for r in top]

    fig_heat = go.Figure(go.Heatmap(
        z=[s1,s2,s3], x=codes, y=['S1','S2','S3'],
        colorscale=[[0,'#22dd66'],[0.5,'#ffaa00'],[1,'#ff2222']],
        text=[[f'{v:.3f}' for v in row] for row in [s1,s2,s3]],
        texttemplate='%{text}', textfont=dict(size=10,color='white'),
        hovertemplate='<b>%{x}</b><br>%{y}: %{z:.3f}s<extra></extra>',
        showscale=False,
    ))
    fig_heat.update_layout(**pl(height=200,
        title=dict(text='Sector Heatmap (grün = schnell)',font=dict(size=10,color=C['muted']))))

    r1 = next((r for r in results if r['code']==d1), results[0])
    r2 = next((r for r in results if r['code']==d2), results[1] if len(results)>1 else results[0])

    fig_bars = go.Figure()
    for r, col in [(r1,dc(d1)),(r2,dc(d2))]:
        fig_bars.add_trace(go.Bar(name=r['code'], x=['S1','S2','S3'],
            y=[r['sectors']['S1'],r['sectors']['S2'],r['sectors']['S3']],
            marker_color=col, text=[f"{v:.3f}" for v in [r['sectors']['S1'],r['sectors']['S2'],r['sectors']['S3']]],
            textposition='outside', textfont=dict(size=9)))
    fig_bars.update_layout(**pl(height=260, barmode='group',
        title=dict(text=f'Sector Comparison: {d1} vs {d2}',font=dict(size=10,color=C['muted'])),
        yaxis_title='Sector Time (s)'))

    maxS = {s: max(r['sectors'][s] for r in results)+0.3 for s in ('S1','S2','S3')}
    fig_radar = go.Figure()
    for r, col in [(r1,dc(d1)),(r2,dc(d2))]:
        vals = [maxS[s]-r['sectors'][s] for s in ('S1','S2','S3')] + [maxS['S1']-r['sectors']['S1']]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=['S1','S2','S3','S1'], name=r['code'],
            fill='toself', line_color=col, fillcolor=hex_alpha(col,0.20)))
    fig_radar.update_layout(**{k:v for k,v in pl().items() if k not in ('xaxis','yaxis')}, height=300,
        polar=dict(bgcolor=C['panel'],
            radialaxis=dict(visible=True,gridcolor=C['grid'],showticklabels=False),
            angularaxis=dict(gridcolor=C['grid'],tickfont=dict(color=C['text'],size=13))),
        title=dict(text=f'Sector Radar: {d1} vs {d2}',font=dict(size=10,color=C['muted'])))

    return [
        driver_pills(d1, d2),
        card(shead('Sector Heatmap','▦'), dcc.Graph(figure=fig_heat, config={'displayModeBar':False}),
             style={'marginBottom':'14px'}),
        html.Div([
            card(shead('Sector Bars','▦'), dcc.Graph(figure=fig_bars, config={'displayModeBar':False})),
            card(shead('Sector Radar','◎'), dcc.Graph(figure=fig_radar, config={'displayModeBar':False})),
        ], className='grid-2'),
    ]

def build_car(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    sess = load_session(year, round_num, 'R')

    spd_data = []
    for r in results[:12]:
        code = r['code']
        max_spd = 290 + (12-r['position'])*1.5
        if sess:
            try:
                dl = sess.laps.pick_driver(code)
                fl = dl.pick_fastest()
                t  = fl.get_telemetry()
                if len(t)>0: max_spd = float(t['Speed'].max())
            except: pass
        spd_data.append({'code':code,'speed':max_spd,'pos':r['position']})

    spd_df = sorted(spd_data, key=lambda x: x['speed'])
    fig_speed = go.Figure(go.Bar(
        y=[d['code'] for d in spd_df], x=[d['speed'] for d in spd_df],
        orientation='h', marker_color=[dc(d['code']) for d in spd_df],
        text=[f"{d['speed']:.0f}" for d in spd_df], textposition='outside', textfont=dict(size=9),
        hovertemplate='<b>%{y}</b>  %{x:.0f} km/h<extra></extra>',
    ))
    fig_speed.update_layout(**pl(height=380,
        title=dict(text='Max Speed Trap (km/h)',font=dict(size=10,color=C['muted'])),
        xaxis_title='Speed (km/h)'))

    cats = ['Top Speed','Cornering','Braking','Traction','Aero','ERS']
    fig_perf = go.Figure()
    for code, col in [(d1,dc(d1)),(d2,dc(d2))]:
        rng = np.random.RandomState(int(hashlib.md5(code.encode()).hexdigest()[:8],16))
        base = 7.5
        vals = [max(2,min(10, base+rng.normal(0,0.7))) for _ in cats]+[0]
        vals[-1] = vals[0]
        fig_perf.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]], name=code, fill='toself',
            line_color=col, fillcolor=hex_alpha(col,0.20)))
    fig_perf.update_layout(**{k:v for k,v in pl().items() if k not in ('xaxis','yaxis')}, height=340,
        polar=dict(bgcolor=C['panel'],
            radialaxis=dict(visible=True,range=[0,10],gridcolor=C['grid'],tickfont=dict(color=C['muted'],size=9)),
            angularaxis=dict(gridcolor=C['grid'],tickfont=dict(color=C['text'],size=11))),
        title=dict(text=f'Car Performance Radar: {d1} vs {d2}',font=dict(size=10,color=C['muted'])))

    return html.Div([
        driver_pills(d1, d2),
        html.Div([
            card(shead('Speed Trap','▲'), dcc.Graph(figure=fig_speed, config={'displayModeBar':False})),
            card(shead('Performance Radar','◎'), dcc.Graph(figure=fig_perf, config={'displayModeBar':False})),
        ], className='grid-2'),
    ])

def build_energy(rd, d1, d2, year, round_num, **kwargs):
    total_laps = rd.get('laps') or 52
    laps = list(range(1, total_laps+1))
    c1, c2 = dc(d1), dc(d2)

    def gen(seed):
        rng = np.random.RandomState(seed)
        dep = np.clip(rng.normal(3.5,0.8,len(laps)),0,5).tolist()
        har = np.clip(rng.normal(3.0,0.6,len(laps)),0,5).tolist()
        bal = [h-d for h,d in zip(har,dep)]
        cum = list(np.cumsum(bal))
        return dep,har,bal,cum

    d1s,h1s,b1s,c1s = gen(hash(d1)%2**31)
    d2s,h2s,b2s,c2s = gen(hash(d2)%2**31)

    fig = make_subplots(rows=2,cols=2,
        subplot_titles=['ERS Deploy (MJ)','ERS Harvest (MJ)','Net Balance','Cumulative'],
        vertical_spacing=0.15, horizontal_spacing=0.1)

    for (dep,har,bal,cum,code,col) in [(d1s,h1s,b1s,c1s,d1,c1),(d2s,h2s,b2s,c2s,d2,c2)]:
        show = code==d1
        fig.add_trace(go.Scatter(x=laps,y=dep,name=f'{code} deploy',
            line=dict(color=col,width=2),showlegend=show), row=1,col=1)
        fig.add_trace(go.Scatter(x=laps,y=har,name=f'{code} harvest',
            line=dict(color=col,width=2,dash='dash'),showlegend=show), row=1,col=2)
        fig.add_trace(go.Bar(x=laps,y=bal,name=f'{code} balance',
            marker_color=col,opacity=0.7,showlegend=False), row=2,col=1)
        fig.add_trace(go.Scatter(x=laps,y=cum,name=f'{code} cumul',
            line=dict(color=col,width=2.5),showlegend=False), row=2,col=2)

    for i in range(1,3):
        for j in range(1,3):
            fig.update_xaxes(gridcolor=C['grid'],linecolor=C['border2'],row=i,col=j)
            fig.update_yaxes(gridcolor=C['grid'],linecolor=C['border2'],row=i,col=j)
    fig.update_layout(**{k:v for k,v in pl().items() if k not in ('xaxis','yaxis')}, height=460,
        title=dict(text=f'Energy Management — {d1} vs {d2}',font=dict(size=10,color=C['muted'])))

    eff1 = sum(h1s)/sum(d1s)*100 if sum(d1s)>0 else 0
    eff2 = sum(h2s)/sum(d2s)*100 if sum(d2s)>0 else 0
    return [
        driver_pills(d1, d2),
        html.Div([
            kpi(f'{d1} Deploy', f'{sum(d1s):.1f} MJ', c1),
            kpi(f'{d2} Deploy', f'{sum(d2s):.1f} MJ', c2),
            kpi(f'{d1} Efficiency', f'{eff1:.1f}%', c1),
            kpi(f'{d2} Efficiency', f'{eff2:.1f}%', c2),
        ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'14px'}),
        card(shead('ERS Analysis','↯'), dcc.Graph(figure=fig, config={'displayModeBar':False})),
    ]

def build_strategy(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    lap_times = rd.get('lap_times',{})
    total_laps = rd.get('laps') or 52
    TCOL = {'S':'#ff3333','M':'#ffcc00','H':'#cccccc','I':'#22aaff','W':'#0088ff'}

    def parse_strat(s, total):
        cs = s.split('-')
        stints, start = [], 1
        for i,c in enumerate(cs):
            end = start+(total//len(cs))-1 if i<len(cs)-1 else total
            stints.append({'compound':c,'start':start,'end':end,'laps':end-start+1})
            start = end+1
        return stints

    fig_strat = go.Figure()
    for i,r in enumerate(results[:10]):
        stints = parse_strat(r['tyre_strategy'], total_laps)
        for stint in stints:
            c = stint['compound']
            fig_strat.add_trace(go.Bar(
                x=[stint['laps']], y=[r['code']], base=[stint['start']-1],
                orientation='h', marker_color=TCOL.get(c,'#888'),
                name=c, showlegend=i==0,
                text=c, textposition='inside',
                hovertemplate=f"<b>{r['code']}</b><br>{c}: L{stint['start']}–{stint['end']}<extra></extra>",
            ))
    fig_strat.update_layout(**{k:v for k,v in pl().items() if k not in ('xaxis',)}, height=360,
        barmode='stack',
        xaxis=dict(range=[0,total_laps],title='Lap',gridcolor=C['grid'],linecolor=C['border2']),
        title=dict(text='Tyre Strategy — Race Stints',font=dict(size=10,color=C['muted'])))

    fig_deg = go.Figure()
    for code, col in [(d1,dc(d1)),(d2,dc(d2))]:
        times = pd.Series([t for t in lap_times.get(code,[]) if 85<=t<=96])
        if len(times)>3:
            x = list(range(1,len(times)+1))
            coeffs = np.polyfit(x,times.tolist(),1)
            trend = [coeffs[0]*i+coeffs[1] for i in x]
            fig_deg.add_trace(go.Scatter(x=x,y=times.tolist(),mode='lines',name=f'{code} raw',
                line=dict(color=col,width=1.5,dash='dot'),opacity=0.6))
            fig_deg.add_trace(go.Scatter(x=x,y=trend,mode='lines',name=f'{code} trend',
                line=dict(color=col,width=2.5)))
    fig_deg.update_layout(**pl(height=260,
        title=dict(text=f'Tyre Degradation — {d1} vs {d2}',font=dict(size=10,color=C['muted'])),
        xaxis_title='Tyre Age (laps)', yaxis_title='Lap Time (s)'))

    ccounts = {}
    for r in results:
        for c in r['tyre_strategy'].split('-'):
            ccounts[c] = ccounts.get(c,0)+1
    fig_comp = go.Figure(go.Pie(
        labels=[f"{'Soft' if k=='S' else 'Medium' if k=='M' else 'Hard' if k=='H' else k} ({k})" for k in ccounts],
        values=list(ccounts.values()),
        marker_colors=[TCOL.get(k,'#888') for k in ccounts],
        hole=0.5, textinfo='label+percent', textfont_size=11,
        hovertemplate='%{label}: %{value}<extra></extra>',
    ))
    fig_comp.update_layout(**pl(height=260,showlegend=False,
        title=dict(text='Compound Usage',font=dict(size=10,color=C['muted']))))

    return [
        driver_pills(d1, d2),
        card(shead('Strategy Timeline','◍'), dcc.Graph(figure=fig_strat, config={'displayModeBar':False}),
             style={'marginBottom':'14px'}),
        html.Div([
            card(shead('Tyre Degradation','▼'), dcc.Graph(figure=fig_deg, config={'displayModeBar':False})),
            card(shead('Compound Usage','◔'), dcc.Graph(figure=fig_comp, config={'displayModeBar':False})),
        ], className='grid-2'),
    ]

def build_track(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    total_laps = rd.get('laps') or 52

    sess = load_session(year, round_num, 'R')
    segments, dist_vals, dom_colors = 40, [], []
    dom_vals = []
    if sess:
        try:
            def get_spd(code):
                return sess.laps.pick_driver(code).pick_fastest().get_telemetry().add_distance()
            tel1 = get_spd(d1); tel2 = get_spd(d2)
            seg = len(tel1)//40
            for i in range(40):
                s,e = i*seg,(i+1)*seg
                a1 = float(tel1['Speed'].iloc[s:e].mean())
                a2 = float(tel2['Speed'].iloc[s:e].mean())
                dom_vals.append(a1-a2)
                dist_vals.append(float(tel1['Distance'].iloc[s]))
        except Exception:
            dom_vals = [np.random.normal(0,5) for _ in range(40)]
            dist_vals = list(range(0,40*130,130))
    else:
        dom_vals = [np.random.normal(0,5) for _ in range(40)]
        dist_vals = list(range(0,40*130,130))

    dom_colors = [dc(d1) if v>0 else dc(d2) for v in dom_vals]
    d1_segs = sum(1 for v in dom_vals if v>0)
    d2_segs = 40-d1_segs

    fig_dom = go.Figure()
    fig_dom.add_hline(y=0, line_color=C['muted'], line_width=1.5)
    fig_dom.add_trace(go.Bar(x=dist_vals, y=dom_vals, marker_color=dom_colors,
        hovertemplate='%{x:.0f}m  Δ %{y:+.1f} km/h<extra></extra>'))
    fig_dom.add_annotation(x=0.02,y=0.95,xref='paper',yref='paper',showarrow=False,
        text=f'{d1}: {d1_segs}/40 ({d1_segs/40*100:.0f}%)',
        font=dict(color=dc(d1),size=11),bgcolor=C['card'],borderpad=4)
    fig_dom.add_annotation(x=0.02,y=0.82,xref='paper',yref='paper',showarrow=False,
        text=f'{d2}: {d2_segs}/40 ({d2_segs/40*100:.0f}%)',
        font=dict(color=dc(d2),size=11),bgcolor=C['card'],borderpad=4)
    fig_dom.update_layout(**pl(height=320,
        title=dict(text=f'Track Dominance — {d1} vs {d2}',font=dict(size=10,color=C['muted'])),
        xaxis_title='Track Distance (m)', yaxis_title='Speed Advantage (km/h)'))

    race_laps = list(range(1,total_laps+1))
    fig_pos = go.Figure()
    for r in results[:6]:
        code, final = r['code'], r['position'] or 20
        rng = np.random.RandomState(hash(code)%2**31)
        start = rng.randint(1,8)
        pos = [start]
        for lap in range(1,len(race_laps)):
            d = rng.choice([-1,0,0,0,1],p=[0.08,0.5,0.2,0.12,0.1])
            if lap > len(race_laps)*0.92:
                d = 1 if pos[-1]>final else -1
            pos.append(max(1,min(20,pos[-1]+d)))
        pos[-1] = final
        fig_pos.add_trace(go.Scatter(x=race_laps,y=pos[:len(race_laps)],name=code,mode='lines',
            line=dict(color=dc(code),width=2),
            hovertemplate=f'<b>{code}</b> Lap %{{x}}: P%{{y}}<extra></extra>'))
    fig_pos.update_layout(**{k:v for k,v in pl().items() if k!='yaxis'}, height=300,
        title=dict(text='Race Position History',font=dict(size=10,color=C['muted'])),
        xaxis_title='Lap',
        yaxis=dict(autorange='reversed',dtick=2,gridcolor=C['grid'],linecolor=C['border2'],title='Position'))

    return [
        driver_pills(d1, d2),
        card(shead('Track Dominance','◌'), dcc.Graph(figure=fig_dom, config={'displayModeBar':False}),
             style={'marginBottom':'14px'}),
        card(shead('Race Position History','▦'), dcc.Graph(figure=fig_pos, config={'displayModeBar':True})),
    ]

def build_standings(rd, d1, d2, year, round_num, **kwargs):
    live_drv = get_live_standings(year)
    live_con = get_live_constructor_standings(year)
    demo_stnd = DEMO['standings']

    drv_list = (live_drv or {}).get('drivers') or demo_stnd['drivers']
    con_list  = live_con or demo_stnd['constructors']

    source_note = ("[OK] Live-Daten von Jolpica/Ergast" if live_drv
                   else "[WARNING] Demo-Daten (API nicht erreichbar)")

    fig_drv = go.Figure(go.Bar(
        y=[d['code'] for d in drv_list[:15]],
        x=[d['points'] for d in drv_list[:15]],
        orientation='h', marker_color=[dc(d['code']) for d in drv_list[:15]],
        text=[str(d['points']) for d in drv_list[:15]], textposition='outside', textfont=dict(size=9),
        hovertemplate='<b>%{y}</b>  %{x} pts<extra></extra>',
    ))
    fig_drv.update_layout(**pl(height=460,
        title=dict(text='Driver Championship',font=dict(size=10,color=C['muted'])),
        xaxis_title='Points'))
    fig_drv.update_yaxes(autorange='reversed')

    fig_con = go.Figure(go.Bar(
        y=[c['team'] for c in con_list],
        x=[c['points'] for c in con_list],
        orientation='h',
        marker_color=[tc(c['team']) for c in con_list],
        text=[str(c['points']) for c in con_list], textposition='outside', textfont=dict(size=9),
        hovertemplate='<b>%{y}</b>  %{x} pts<extra></extra>',
    ))
    fig_con.update_layout(**pl(height=360,
        title=dict(text='Constructor Championship',font=dict(size=10,color=C['muted'])),
        xaxis_title='Points'))
    fig_con.update_yaxes(autorange='reversed')

    top5 = [d['code'] for d in drv_list[:5]]
    race_names = [r['name'].replace(' Grand Prix',' GP') for r in DEMO['races']]
    running = {c:0 for c in top5}
    cumpts = {c:[] for c in top5}
    for r in DEMO['races']:
        for code in top5:
            res = next((x for x in r.get('results',[]) if x['code']==code), None)
            running[code] += res['points'] if res else 0
            cumpts[code].append(running[code])
    fig_prog = go.Figure()
    for code in top5:
        col = dc(code)
        fig_prog.add_trace(go.Scatter(x=race_names, y=cumpts[code], name=code, mode='lines+markers',
            line=dict(color=col,width=2.5), marker=dict(size=6,color=col),
            hovertemplate=f'<b>{code}</b>  %{{x}}: %{{y}} pts<extra></extra>'))
    fig_prog.update_layout(**pl(height=280,
        title=dict(text='Championship Progression (Top 5)',font=dict(size=10,color=C['muted'])),
        xaxis_title='Race', yaxis_title='Points'))
    fig_prog.update_xaxes(tickangle=45, tickfont=dict(size=8))

    return [
        html.Div(source_note, style={'fontSize':'11px','color':C['muted'],'marginBottom':'12px',
            'background':C['panel'],'border':f'1px solid {C["border"]}','borderRadius':'6px','padding':'8px 12px'}),
        card(shead('Season Progression','▲'), dcc.Graph(figure=fig_prog, config={'displayModeBar':False}),
             style={'marginBottom':'14px'}),
        html.Div([
            card(shead('Driver Standings','★'), dcc.Graph(figure=fig_drv, config={'displayModeBar':False})),
            card(shead('Constructor Standings','▣'), dcc.Graph(figure=fig_con, config={'displayModeBar':False})),
        ], className='grid-2'),
    ]

def build_live(rd, d1, d2, year, round_num, **kwargs):
    results = rd.get('results',[])
    TCOL = {'S':'#ff3333','M':'#ffcc00','H':'#cccccc','I':'#22aaff','W':'#0088ff'}

    rows = []
    for r in results:
        col = dc(r['code'])
        gap = f"+{r['gap']:.3f}" if r['gap']>0 else 'LEADER'
        last_comp = r['tyre_strategy'].split('-')[-1]
        rows.append(html.Tr([
            html.Td(f"P{r['position']}", style={'color':C['accent'] if r['position']==1 else C['muted'],'fontWeight':'700','width':'36px','fontSize':'13px'}),
            html.Td(html.Span('●', style={'color':col,'fontSize':'15px','marginRight':'5px'})),
            html.Td(r['code'], style={'fontWeight':'700','fontSize':'13px','color':col}),
            html.Td(r['team'][:12], style={'color':C['muted'],'fontSize':'10px','minWidth':'85px'}),
            html.Td(gap, style={'fontFamily':'monospace','fontSize':'11px','color':'#22dd88' if r['position']==1 else C['text']}),
            html.Td(lap_str(r['fastest_lap']), style={'fontFamily':'monospace','fontSize':'11px','color':C['muted']}),
            html.Td(last_comp, style={'color':TCOL.get(last_comp,'#888'),'fontWeight':'700','fontSize':'12px'}),
        ], style={'borderBottom':f'1px solid {C["border"]}','height':'30px'}))

    code_block = lambda s: html.Code(s, style={
        'display':'block','background':C['bg'],'border':f'1px solid {C["border"]}',
        'borderRadius':'6px','padding':'10px 14px','fontFamily':'monospace',
        'fontSize':'11px','color':'#22ee88','marginBottom':'8px','whiteSpace':'pre-wrap'})

    return [
        html.Div('[WARNING] Live-Daten erfordern einen aktiven F1-Session-Livestream. Die Tabelle zeigt die letzten geladenen Daten.',
            style={'background':'rgba(232,0,45,0.08)','border':'1px solid rgba(232,0,45,0.25)',
                   'borderRadius':'6px','padding':'10px 14px','fontSize':'12px','color':'#ff8888','marginBottom':'14px'}),
        card(shead('Timing Tower','●'),
            html.Table([
                html.Thead(html.Tr([html.Th(h) for h in ['Pos','●','Drv','Team','Gap','Best Lap','Tyre']])),
                html.Tbody(rows),
            ], className='f1-table'),
        style={'marginBottom':'14px'}),
        card(shead('Live Recording starten','≋'),
            html.Div([
                html.P("Terminal 1 — während Rennen läuft:", style={'fontSize':'12px','color':C['muted'],'marginBottom':'6px'}),
                code_block("python -m fastf1.livetiming.client live_data.txt"),
                html.P("Terminal 2 — Dashboard:", style={'fontSize':'12px','color':C['muted'],'marginBottom':'6px','marginTop':'8px'}),
                code_block("python f1_dashboard.py"),
                html.P("In Python mit Live-Daten:", style={'fontSize':'12px','color':C['muted'],'marginBottom':'6px','marginTop':'8px'}),
                code_block("""import fastf1
from fastf1.livetiming import LiveTimingData
session = fastf1.get_session(2026, 'British Grand Prix', 'R')
session.load(livedata=LiveTimingData('live_data.txt'))"""),
            ])),
    ]

# ══════════════════════════════════════════════════════════════════════════════
# ── ML PREDICTIONS TAB ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Full 2026 calendar with track characteristics
TRACK_DATA = {
  1: {
    "round":1,"name":"Australian GP","circuit":"Albert Park",
    "country":"Australia","date":"2026-03-08","laps":58,
    "completed":True,"length_km":5.278,"corners":14,
    "drs_zones":3,"lap_record":"1:20.235 (Pérez, 2023)",
    "first_gp":1996,"layout_type":"street",
    "characteristics":["High-speed","Safety Car prone","Street circuit"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★","McLaren":"★★★","Red Bull":"★★"},
    "profile":{"power":0.65,"downforce":0.70},
    "svg_path":"M 100,48 L 210,48 C 228,48 240,62 238,80 L 234,120 C 232,138 245,150 258,155 L 265,185 C 268,202 258,215 242,215 L 215,212 C 198,210 186,222 184,240 L 180,258 C 176,272 162,278 148,274 L 72,265 C 54,260 44,244 48,226 L 52,85 C 54,64 70,48 90,48 Z",
    "sector_markers":[{"x":170,"y":50},{"x":255,"y":185},{"x":100,"y":260}],
    "drs_svg":["M 100,50 L 200,50","M 50,150 L 50,220"]
  },
  2: {
    "round":2,"name":"Chinese GP","circuit":"Shanghai",
    "country":"China","date":"2026-03-15","laps":56,
    "completed":True,"length_km":5.451,"corners":16,
    "drs_zones":2,"lap_record":"1:32.238 (Schumacher, 2004)",
    "first_gp":2004,"layout_type":"permanent",
    "characteristics":["Long straights","Heavy braking","Overtaking"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★","McLaren":"★★★","Red Bull":"★★"},
    "profile":{"power":0.70,"downforce":0.60},
    "svg_path":"M 85,52 L 185,52 C 205,52 218,66 215,86 L 210,125 C 208,145 222,158 238,158 L 255,158 C 270,158 278,172 272,188 L 258,225 C 250,245 230,255 210,250 L 190,245 C 170,240 158,255 156,275 L 155,283 C 153,293 142,298 130,294 C 118,290 113,278 116,266 L 121,236 C 125,216 111,203 93,203 L 70,202 C 50,201 38,186 42,168 L 48,80 C 50,62 66,50 85,52 Z",
    "sector_markers":[{"x":155,"y":52},{"x":268,"y":172},{"x":110,"y":270}],
    "drs_svg":["M 87,52 L 183,52","M 42,168 L 42,200"]
  },
  3: {
    "round":3,"name":"Japanese GP","circuit":"Suzuka",
    "country":"Japan","date":"2026-03-29","laps":53,
    "completed":True,"length_km":5.807,"corners":18,
    "drs_zones":1,"lap_record":"1:30.983 (Hamilton, 2019)",
    "first_gp":1987,"layout_type":"permanent",
    "characteristics":["Figure-8 layout","High-speed","Aero sensitive"],
    "track_strengths":{"Mercedes":"★★★","Red Bull":"★★★","Ferrari":"★★","McLaren":"★★"},
    "profile":{"power":0.60,"downforce":0.80},
    "svg_path":"M 148,35 C 170,33 184,50 180,70 L 174,95 C 170,115 182,130 198,130 C 214,130 225,118 222,102 L 218,75 C 215,58 228,45 245,48 C 262,51 270,68 265,84 L 255,135 C 248,158 228,168 208,162 L 178,155 C 158,148 145,160 142,180 L 138,215 C 134,238 118,252 98,248 C 78,244 66,226 70,206 L 76,165 C 80,145 68,130 50,128 L 38,100 C 32,80 46,62 65,58 L 120,38 Z",
    "sector_markers":[{"x":210,"y":70},{"x":148,"y":155},{"x":75,"y":215}],
    "drs_svg":["M 65,58 L 148,38"]
  },
  4: {
    "round":4,"name":"Bahrain GP","circuit":"Sakhir",
    "country":"Bahrain","date":"2026-04-10","laps":57,
    "completed":False,"length_km":5.412,"corners":15,
    "drs_zones":3,"lap_record":"1:31.447 (de la Rosa, 2005)",
    "first_gp":2004,"layout_type":"permanent",
    "characteristics":["Night race","Desert heat","Tyre management"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Ferrari":"★★","Red Bull":"★★"},
    "profile":{"power":0.72,"downforce":0.62},
    "svg_path":"M 90,50 L 200,50 C 220,50 232,66 228,86 L 222,118 C 218,138 232,152 248,155 L 256,185 C 260,205 246,220 228,215 L 205,210 C 186,205 174,220 170,242 C 166,262 148,272 130,265 L 58,250 C 40,242 30,224 34,204 L 38,82 C 40,62 58,48 78,50 Z",
    "sector_markers":[{"x":155,"y":50},{"x":250,"y":185},{"x":80,"y":255}],
    "drs_svg":["M 90,50 L 198,50","M 228,215 L 170,242","M 38,204 L 38,82"]
  },
  5: {
    "round":5,"name":"Saudi Arabian GP","circuit":"Jeddah",
    "country":"Saudi Arabia","date":"2026-04-17","laps":50,
    "completed":False,"length_km":6.174,"corners":27,
    "drs_zones":3,"lap_record":"1:30.734 (Hamilton, 2021)",
    "first_gp":2021,"layout_type":"street",
    "characteristics":["Fastest street circuit","Night race","27 corners"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Ferrari":"★★","Red Bull":"★★"},
    "profile":{"power":0.88,"downforce":0.40},
    "svg_path":"M 68,262 L 238,262 C 255,262 265,248 260,230 L 252,195 C 246,178 258,164 270,158 L 270,100 C 270,80 255,68 238,72 L 215,76 C 198,80 188,68 190,50 C 192,32 176,22 158,28 L 52,50 C 34,56 26,74 32,92 L 40,225 C 44,248 54,264 68,262 Z",
    "sector_markers":[{"x":158,"y":50},{"x":268,"y":160},{"x":55,"y":155}],
    "drs_svg":["M 68,262 L 238,262","M 52,50 L 158,28","M 270,100 L 270,158"]
  },
  6: {
    "round":6,"name":"Miami GP","circuit":"Miami",
    "country":"USA","date":"2026-05-01","laps":57,
    "completed":True,"length_km":5.412,"corners":19,
    "drs_zones":3,"lap_record":"1:29.708 (Verstappen, 2023)",
    "first_gp":2022,"layout_type":"street",
    "characteristics":["Sprint weekend","Stadium section","Hot & humid"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Ferrari":"★★","Red Bull":"★★"},
    "profile":{"power":0.65,"downforce":0.65},
    "svg_path":"M 92,50 L 215,50 C 232,50 243,64 240,82 L 235,110 C 232,128 244,142 258,145 L 268,175 C 272,192 260,205 244,202 L 220,198 C 202,194 192,208 190,228 L 186,252 C 182,268 166,276 150,270 L 68,258 C 50,252 40,234 44,215 L 48,82 C 50,62 68,48 88,50 Z",
    "sector_markers":[{"x":165,"y":50},{"x":262,"y":175},{"x":70,"y":255}],
    "drs_svg":["M 92,50 L 213,50","M 244,202 L 190,228","M 44,215 L 44,82"]
  },
  7: {
    "round":7,"name":"Canadian GP","circuit":"Montreal",
    "country":"Canada","date":"2026-05-22","laps":70,
    "completed":True,"length_km":4.361,"corners":14,
    "drs_zones":3,"lap_record":"1:13.078 (Bottas, 2019)",
    "first_gp":1978,"layout_type":"street",
    "characteristics":["Wall of Champions","Safety Car likely","Low downforce"],
    "track_strengths":{"Ferrari":"★★★","Mercedes":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.80,"downforce":0.50},
    "svg_path":"M 105,45 L 205,45 C 224,45 236,60 232,78 L 226,112 C 222,132 235,146 250,148 L 262,180 C 266,198 254,212 238,210 L 218,206 C 200,202 188,218 185,238 C 182,258 165,268 148,262 L 62,248 C 44,242 34,224 38,205 L 42,80 C 44,60 62,44 82,45 Z",
    "sector_markers":[{"x":160,"y":45},{"x":255,"y":180},{"x":65,"y":248}],
    "drs_svg":["M 105,45 L 203,45","M 42,205 L 42,80","M 238,210 L 185,238"]
  },
  8: {
    "round":8,"name":"Monaco GP","circuit":"Monte Carlo",
    "country":"Monaco","date":"2026-06-05","laps":78,
    "completed":True,"length_km":3.337,"corners":19,
    "drs_zones":1,"lap_record":"1:12.909 (Hamilton, 2021)",
    "first_gp":1950,"layout_type":"street",
    "characteristics":["No overtaking","Qualifying key","Tunnel section","78 laps"],
    "track_strengths":{"Ferrari":"★★★","Red Bull":"★★","Mercedes":"★★","McLaren":"★★"},
    "profile":{"power":0.25,"downforce":0.95},
    "svg_path":"M 158,258 C 170,258 180,248 182,236 L 186,175 C 188,158 200,150 215,153 L 232,156 C 246,159 254,148 250,134 L 240,82 C 236,66 222,60 206,63 L 138,68 C 122,70 112,60 108,46 L 103,30 C 99,18 85,14 72,18 L 46,36 C 32,48 30,64 36,78 L 44,198 C 46,220 56,238 74,245 L 138,260 Z",
    "sector_markers":[{"x":180,"y":100},{"x":182,"y":200},{"x":90,"y":240}],
    "drs_svg":["M 36,78 L 44,140"]
  },
  9: {
    "round":9,"name":"Spanish GP (Barcelona)","circuit":"Barcelona-Catalunya",
    "country":"Spain","date":"2026-06-12","laps":66,
    "completed":True,"length_km":4.675,"corners":16,
    "drs_zones":2,"lap_record":"1:16.330 (Verstappen, 2023)",
    "first_gp":1991,"layout_type":"permanent",
    "characteristics":["High tyre wear","Technical","Pre-season test venue"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★","McLaren":"★★★","Red Bull":"★★"},
    "profile":{"power":0.60,"downforce":0.80},
    "svg_path":"M 80,50 L 220,50 C 240,50 252,64 250,84 L 245,112 C 242,132 255,146 258,164 C 260,182 248,196 232,190 L 216,184 C 200,178 184,188 180,204 L 176,238 C 172,256 156,264 140,258 L 72,248 C 54,242 44,224 48,204 L 52,82 C 54,62 68,48 80,50 Z",
    "sector_markers":[{"x":155,"y":50},{"x":252,"y":164},{"x":75,"y":248}],
    "drs_svg":["M 80,50 L 218,50","M 48,204 L 48,82"]
  },
  10: {
    "round":10,"name":"Austrian GP","circuit":"Red Bull Ring",
    "country":"Austria","date":"2026-06-26","laps":71,
    "completed":True,"length_km":4.318,"corners":10,
    "drs_zones":3,"lap_record":"1:05.619 (Sainz, 2020)",
    "first_gp":1970,"layout_type":"permanent",
    "characteristics":["Sprint weekend","Short lap","Mountain setting"],
    "track_strengths":{"Mercedes":"★★★","Red Bull":"★★★","McLaren":"★★","Ferrari":"★★"},
    "profile":{"power":0.70,"downforce":0.65},
    "svg_path":"M 138,55 C 162,50 180,66 176,88 L 170,120 C 166,142 180,158 198,158 L 225,158 C 248,158 260,175 252,198 L 238,232 C 228,255 205,264 182,255 L 88,238 C 64,228 52,204 60,180 L 68,142 C 74,118 58,100 50,82 C 44,65 58,50 75,52 L 115,54 Z",
    "sector_markers":[{"x":165,"y":56},{"x":248,"y":175},{"x":65,"y":238}],
    "drs_svg":["M 115,54 L 176,88","M 198,158 L 252,198","M 60,180 L 60,82"]
  },
  11: {
    "round":11,"name":"British GP","circuit":"Silverstone",
    "country":"UK","date":"2026-07-03","laps":52,
    "completed":True,"length_km":5.891,"corners":18,
    "drs_zones":2,"lap_record":"1:27.097 (Verstappen, 2020)",
    "first_gp":1950,"layout_type":"permanent",
    "characteristics":["High-speed","Maggotts-Becketts","Home of F1"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Ferrari":"★★","Red Bull":"★★"},
    "profile":{"power":0.72,"downforce":0.75},
    "svg_path":"M 75,52 L 222,52 C 244,52 256,66 253,88 L 248,115 C 246,130 258,144 258,162 C 258,180 244,192 228,188 L 212,182 C 196,175 180,184 177,198 L 172,234 C 168,252 152,262 134,258 L 72,252 C 50,248 38,232 40,210 L 42,82 C 42,62 57,50 75,52 Z",
    "sector_markers":[{"x":155,"y":52},{"x":252,"y":162},{"x":80,"y":250}],
    "drs_svg":["M 75,52 L 220,52","M 40,210 L 40,82"]
  },
  12: {
    "round":12,"name":"Belgian GP","circuit":"Spa-Francorchamps",
    "country":"Belgium","date":"2026-07-17","laps":44,
    "completed":False,"length_km":7.004,"corners":19,
    "drs_zones":2,"lap_record":"1:46.286 (Bottas, 2018)",
    "first_gp":1950,"layout_type":"permanent",
    "characteristics":["Longest circuit","Eau Rouge","Weather variable","Power sensitive"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.88,"downforce":0.50},
    "svg_path":"M 220,45 C 242,45 258,60 255,80 C 252,100 235,108 215,110 L 95,198 C 78,214 75,228 84,242 L 90,252 C 96,262 90,272 78,275 L 52,278 C 36,280 28,268 32,252 L 38,75 C 40,55 56,42 76,43 Z",
    "sector_markers":[{"x":155,"y":65},{"x":85,"y":200},{"x":45,"y":265}],
    "drs_svg":["M 76,43 L 220,45","M 38,75 L 38,200"]
  },
  13: {
    "round":13,"name":"Hungarian GP","circuit":"Hungaroring",
    "country":"Hungary","date":"2026-07-24","laps":70,
    "completed":False,"length_km":4.381,"corners":14,
    "drs_zones":2,"lap_record":"1:16.627 (Hamilton, 2020)",
    "first_gp":1986,"layout_type":"permanent",
    "characteristics":["Twisty","Downforce heavy","Overtaking difficult","Very hot"],
    "track_strengths":{"Ferrari":"★★★","Mercedes":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.30,"downforce":0.92},
    "svg_path":"M 152,42 C 174,40 188,54 185,74 L 180,96 C 177,112 188,126 205,128 L 226,130 C 242,132 250,148 244,164 L 233,184 C 225,200 230,216 244,226 L 252,236 C 260,248 255,262 240,265 L 78,268 C 60,270 48,254 50,236 L 53,188 C 54,170 42,157 34,144 L 32,104 C 30,85 44,70 62,68 L 122,44 Z",
    "sector_markers":[{"x":180,"y":50},{"x":248,"y":200},{"x":55,"y":258}],
    "drs_svg":["M 122,44 L 185,74","M 50,236 L 50,188"]
  },
  14: {
    "round":14,"name":"Dutch GP","circuit":"Zandvoort",
    "country":"Netherlands","date":"2026-08-21","laps":72,
    "completed":False,"length_km":4.259,"corners":14,
    "drs_zones":2,"lap_record":"1:11.097 (Hamilton, 2021)",
    "first_gp":1952,"layout_type":"permanent",
    "characteristics":["Banked corners","Compact","North Sea winds"],
    "track_strengths":{"Mercedes":"★★★","Red Bull":"★★★","McLaren":"★★","Ferrari":"★★"},
    "profile":{"power":0.45,"downforce":0.88},
    "svg_path":"M 148,36 C 172,34 188,50 185,72 L 180,98 C 177,115 188,128 204,130 L 222,132 C 240,132 250,148 246,165 L 238,192 C 230,212 216,224 198,220 L 174,215 C 156,210 145,225 145,245 C 145,264 128,275 110,267 C 92,260 86,242 93,224 L 100,190 C 106,168 93,154 76,152 L 58,150 C 40,148 30,133 34,115 L 42,70 C 46,50 64,36 86,35 Z",
    "sector_markers":[{"x":168,"y":38},{"x":244,"y":160},{"x":90,"y":225}],
    "drs_svg":["M 86,35 L 185,72","M 34,115 L 34,150"]
  },
  15: {
    "round":15,"name":"Italian GP","circuit":"Monza",
    "country":"Italy","date":"2026-09-04","laps":53,
    "completed":False,"length_km":5.793,"corners":11,
    "drs_zones":2,"lap_record":"1:21.046 (Barrichello, 2004)",
    "first_gp":1950,"layout_type":"permanent",
    "characteristics":["Temple of Speed","360 km/h top speed","Slipstreaming","Low downforce"],
    "track_strengths":{"Ferrari":"★★★","Mercedes":"★★★","McLaren":"★★★","Red Bull":"★★"},
    "profile":{"power":0.95,"downforce":0.20},
    "svg_path":"M 58,255 L 245,255 C 258,255 265,242 260,228 L 253,212 C 248,198 255,186 264,182 L 268,138 C 270,120 260,108 246,112 L 230,116 C 216,120 208,108 212,94 L 218,52 C 222,36 210,26 196,30 L 70,32 C 52,34 42,48 44,65 L 50,208 C 50,232 54,252 58,255 Z",
    "sector_markers":[{"x":155,"y":32},{"x":265,"y":160},{"x":52,"y":230}],
    "drs_svg":["M 70,32 L 218,52","M 50,208 L 50,65"]
  },
  16: {
    "round":16,"name":"Spanish GP (Madrid)","circuit":"Circuit de Madrid",
    "country":"Spain","date":"2026-09-11","laps":57,
    "completed":False,"length_km":5.470,"corners":20,
    "drs_zones":3,"lap_record":"N/A (Inaugural race)",
    "first_gp":2026,"layout_type":"street",
    "characteristics":["Brand new circuit","Sprint weekend","Urban layout","20 corners"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.60,"downforce":0.75},
    "svg_path":"M 98,48 L 208,48 C 228,48 240,64 236,84 L 230,118 C 226,138 240,152 255,155 L 262,185 C 265,205 252,218 236,215 L 212,210 C 194,206 182,220 180,242 L 176,260 C 172,276 156,282 140,276 L 62,262 C 44,256 34,238 38,218 L 42,82 C 44,62 62,46 82,48 Z",
    "sector_markers":[{"x":155,"y":48},{"x":258,"y":185},{"x":75,"y":260}],
    "drs_svg":["M 98,48 L 206,48","M 38,218 L 38,82","M 236,215 L 180,242"]
  },
  17: {
    "round":17,"name":"Azerbaijan GP","circuit":"Baku City Circuit",
    "country":"Azerbaijan","date":"2026-09-24","laps":51,
    "completed":False,"length_km":6.003,"corners":20,
    "drs_zones":3,"lap_record":"1:43.009 (Leclerc, 2019)",
    "first_gp":2016,"layout_type":"street",
    "characteristics":["Longest straight in F1","Castle section 7.6m wide","Safety Car certain","350 km/h"],
    "track_strengths":{"Ferrari":"★★★","Mercedes":"★★","McLaren":"★★","Red Bull":"★★★"},
    "profile":{"power":0.85,"downforce":0.45},
    "svg_path":"M 68,262 L 238,262 C 256,262 266,246 260,228 L 252,195 C 246,178 258,164 270,158 L 270,100 C 270,80 255,68 238,72 L 215,76 C 198,80 188,68 190,50 C 192,32 175,22 158,28 L 52,50 C 34,56 26,74 32,92 L 40,225 C 44,248 54,264 68,262 Z",
    "sector_markers":[{"x":158,"y":35},{"x":268,"y":160},{"x":38,"y":160}],
    "drs_svg":["M 68,262 L 238,262","M 52,50 L 158,28","M 270,100 L 270,158"]
  },
  18: {
    "round":18,"name":"Singapore GP","circuit":"Marina Bay",
    "country":"Singapore","date":"2026-10-09","laps":61,
    "completed":False,"length_km":5.063,"corners":23,
    "drs_zones":3,"lap_record":"1:35.867 (Hamilton, 2023)",
    "first_gp":2008,"layout_type":"street",
    "characteristics":["Night race","Most physically demanding","23 corners","High humidity"],
    "track_strengths":{"Ferrari":"★★★","Mercedes":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.35,"downforce":0.90},
    "svg_path":"M 152,255 C 166,255 177,244 178,230 L 180,188 C 181,172 194,163 210,166 L 230,170 C 246,174 256,162 252,146 L 244,105 C 238,88 224,80 208,84 L 185,88 C 168,92 157,80 154,64 L 150,45 C 147,30 133,24 120,30 L 50,56 C 34,64 28,82 34,98 L 40,188 C 42,215 52,240 70,250 L 130,257 Z",
    "sector_markers":[{"x":195,"y":90},{"x":245,"y":170},{"x":50,"y":175}],
    "drs_svg":["M 120,30 L 154,64","M 34,98 L 34,188","M 210,166 L 252,146"]
  },
  19: {
    "round":19,"name":"US GP","circuit":"Austin (COTA)",
    "country":"USA","date":"2026-10-23","laps":56,
    "completed":False,"length_km":5.513,"corners":20,
    "drs_zones":2,"lap_record":"1:36.169 (Leclerc, 2019)",
    "first_gp":2012,"layout_type":"permanent",
    "characteristics":["Sprint weekend","41m elevation change","Esses complex","Stadium section"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Red Bull":"★★★","Ferrari":"★★"},
    "profile":{"power":0.68,"downforce":0.72},
    "svg_path":"M 80,50 C 58,50 44,66 46,88 L 50,125 C 52,145 40,160 35,178 C 30,196 38,215 56,220 L 90,228 C 108,232 115,248 110,265 C 106,278 118,288 132,285 L 225,265 C 245,260 255,242 250,222 L 242,195 C 236,175 248,160 260,155 L 265,120 C 268,100 255,86 238,88 L 215,90 C 198,92 186,78 188,62 C 190,46 175,38 160,44 L 95,50 Z",
    "sector_markers":[{"x":160,"y":44},{"x":258,"y":155},{"x":110,"y":260}],
    "drs_svg":["M 95,50 L 160,44","M 250,222 L 250,155"]
  },
  20: {
    "round":20,"name":"Mexico City GP","circuit":"Hermanos Rodríguez",
    "country":"Mexico","date":"2026-10-30","laps":71,
    "completed":False,"length_km":4.304,"corners":17,
    "drs_zones":3,"lap_record":"1:17.774 (Bottas, 2021)",
    "first_gp":1963,"layout_type":"permanent",
    "characteristics":["2285m altitude","15% less engine power","Foro Sol stadium","Thin air"],
    "track_strengths":{"Red Bull":"★★★","Mercedes":"★★","Ferrari":"★★","McLaren":"★★"},
    "profile":{"power":0.78,"downforce":0.55},
    "svg_path":"M 95,50 L 208,50 C 228,50 240,66 236,86 L 230,120 C 226,140 240,155 256,158 L 264,188 C 268,208 254,222 238,218 L 212,212 C 194,208 182,224 180,245 C 178,265 160,275 142,268 L 60,252 C 42,244 32,225 36,205 L 40,82 C 42,62 60,48 80,50 Z",
    "sector_markers":[{"x":155,"y":50},{"x":260,"y":188},{"x":55,"y":250}],
    "drs_svg":["M 95,50 L 206,50","M 36,205 L 36,82","M 238,218 L 180,245"]
  },
  21: {
    "round":21,"name":"São Paulo GP","circuit":"Interlagos",
    "country":"Brazil","date":"2026-11-06","laps":71,
    "completed":False,"length_km":4.309,"corners":15,
    "drs_zones":2,"lap_record":"1:10.540 (Bottas, 2018)",
    "first_gp":1973,"layout_type":"permanent",
    "characteristics":["Anti-clockwise","Sprint weekend","40m elevation change","Rainy season"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.65,"downforce":0.70},
    "svg_path":"M 162,45 C 185,42 200,58 196,80 L 190,108 C 186,128 172,140 155,136 C 138,132 128,118 132,100 C 136,82 125,68 108,66 L 65,68 C 46,70 35,86 38,106 L 42,178 C 44,200 38,218 28,232 C 18,246 22,264 38,270 L 185,275 C 205,278 218,262 215,242 L 210,205 C 206,185 220,170 238,170",
    "sector_markers":[{"x":178,"y":48},{"x":240,"y":175},{"x":38,"y":200}],
    "drs_svg":["M 108,66 L 196,80","M 38,106 L 38,178"]
  },
  22: {
    "round":22,"name":"Las Vegas GP","circuit":"Las Vegas Strip",
    "country":"USA","date":"2026-11-19","laps":50,
    "completed":False,"length_km":6.120,"corners":17,
    "drs_zones":2,"lap_record":"1:35.490 (Piastri, 2023)",
    "first_gp":2023,"layout_type":"street",
    "characteristics":["1.9km main straight","Night race","Cold asphalt","Neon lights"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★","McLaren":"★★★","Red Bull":"★★"},
    "profile":{"power":0.90,"downforce":0.35},
    "svg_path":"M 70,262 L 240,262 C 258,262 268,246 262,228 L 255,155 C 252,136 264,122 275,115 L 275,82 C 275,64 260,50 242,54 L 218,58 C 200,62 188,50 188,34 C 188,18 170,10 154,18 L 48,48 C 30,56 22,76 28,94 L 35,228 C 38,248 52,264 70,262 Z",
    "sector_markers":[{"x":154,"y":20},{"x":270,"y":115},{"x":38,"y":160}],
    "drs_svg":["M 70,262 L 240,262","M 48,48 L 188,34"]
  },
  23: {
    "round":23,"name":"Qatar GP","circuit":"Lusail",
    "country":"Qatar","date":"2026-11-27","laps":57,
    "completed":False,"length_km":5.380,"corners":16,
    "drs_zones":2,"lap_record":"1:24.319 (Verstappen, 2023)",
    "first_gp":2021,"layout_type":"permanent",
    "characteristics":["High-speed flowing","Dusty offline","Night race","Tyre wear"],
    "track_strengths":{"Mercedes":"★★★","McLaren":"★★★","Ferrari":"★★","Red Bull":"★★"},
    "profile":{"power":0.55,"downforce":0.80},
    "svg_path":"M 145,40 C 168,38 184,55 180,78 L 174,110 C 170,132 182,148 198,148 L 228,148 C 248,148 260,165 255,185 L 245,220 C 238,242 218,254 196,248 L 92,238 C 70,232 56,212 60,190 L 65,148 C 68,126 55,112 40,108 L 35,78 C 32,58 48,42 68,42 L 120,40 Z",
    "sector_markers":[{"x":162,"y":40},{"x":252,"y":175},{"x":55,"y":230}],
    "drs_svg":["M 120,40 L 180,78","M 60,190 L 60,108"]
  },
  24: {
    "round":24,"name":"Abu Dhabi GP","circuit":"Yas Marina",
    "country":"UAE","date":"2026-12-04","laps":58,
    "completed":False,"length_km":5.281,"corners":16,
    "drs_zones":2,"lap_record":"1:26.103 (Verstappen, 2021)",
    "first_gp":2009,"layout_type":"permanent",
    "characteristics":["Season finale","Twilight race","Hotel section","Championship decider"],
    "track_strengths":{"Mercedes":"★★★","Ferrari":"★★★","McLaren":"★★","Red Bull":"★★"},
    "profile":{"power":0.65,"downforce":0.68},
    "svg_path":"M 88,50 L 215,50 C 234,50 246,66 242,86 L 235,125 C 232,145 246,160 260,162 L 268,192 C 272,212 258,226 242,222 L 215,217 C 196,212 184,228 182,248 C 180,268 162,278 144,270 L 64,255 C 46,248 36,228 40,208 L 44,85 C 46,64 64,48 85,50 Z",
    "sector_markers":[{"x":155,"y":50},{"x":264,"y":192},{"x":70,"y":255}],
    "drs_svg":["M 88,50 L 213,50","M 40,208 L 40,85"]
  },
}

# Bahrain (was R4) and Saudi Arabia (was R5) cancelled — remove and renumber
_CANCELLED_ROUNDS = {4, 5}
_renumbered = {}
_new_r = 1
for _old_r in sorted(TRACK_DATA.keys()):
    if _old_r not in _CANCELLED_ROUNDS:
        _e = dict(TRACK_DATA[_old_r])
        _e["round"] = _new_r
        _renumbered[_new_r] = _e
        _new_r += 1
TRACK_DATA = _renumbered

CALENDAR_BY_ROUND = TRACK_DATA

# ── Dummy first entry placeholder (for fallback compatibility) ─────────────────
_TRACK_FALLBACK = TRACK_DATA[10]  # Belgian GP is now R10

RACE_CALENDAR_2026 = list(TRACK_DATA.values())  # kept for any legacy references

# Current 2026 driver standings for ML features
STANDINGS_2026 = {
    "ANT": {"name":"Andrea Kimi Antonelli","team":"Mercedes",    "points":179,"color":"#27F4D2"},
    "RUS": {"name":"George Russell",        "team":"Mercedes",    "points":154,"color":"#27F4D2"},
    "HAM": {"name":"Lewis Hamilton",        "team":"Ferrari",     "points":147,"color":"#E8002D"},
    "LEC": {"name":"Charles Leclerc",       "team":"Ferrari",     "points":108,"color":"#E8002D"},
    "NOR": {"name":"Lando Norris",          "team":"McLaren",     "points":97, "color":"#FF8000"},
    "PIA": {"name":"Oscar Piastri",         "team":"McLaren",     "points":82, "color":"#FF8000"},
    "VER": {"name":"Max Verstappen",        "team":"Red Bull",    "points":76, "color":"#3671C6"},
    "HAD": {"name":"Isack Hadjar",          "team":"Red Bull",    "points":52, "color":"#3671C6"},
    "GAS": {"name":"Pierre Gasly",          "team":"Alpine",      "points":42, "color":"#FF69B4"},
    "LAW": {"name":"Liam Lawson",           "team":"Racing Bulls","points":39, "color":"#6692FF"},
    "LIN": {"name":"Arvid Lindblad",        "team":"Racing Bulls","points":20, "color":"#6692FF"},
    "BEA": {"name":"Oliver Bearman",        "team":"Haas",        "points":18, "color":"#B6BABD"},
    "COL": {"name":"Franco Colapinto",      "team":"Alpine",      "points":18, "color":"#FF69B4"},
    "BOR": {"name":"Gabriel Bortoleto",     "team":"Audi",        "points":6,  "color":"#F50000"},
    "SAI": {"name":"Carlos Sainz",          "team":"Williams",    "points":6,  "color":"#64C4FF"},
    "ALB": {"name":"Alexander Albon",       "team":"Williams",    "points":5,  "color":"#64C4FF"},
    "OCO": {"name":"Esteban Ocon",          "team":"Haas",        "points":3,  "color":"#B6BABD"},
    "ALO": {"name":"Fernando Alonso",       "team":"Aston Martin","points":1,  "color":"#358C75"},
    "HUL": {"name":"Nico Hülkenberg",       "team":"Audi",        "points":0,  "color":"#F50000"},
    "BOT": {"name":"Valtteri Bottas",       "team":"Cadillac",    "points":0,  "color":"#C8A400"},
    "PER": {"name":"Sergio Pérez",          "team":"Cadillac",    "points":0,  "color":"#C8A400"},
    "STR": {"name":"Lance Stroll",          "team":"Aston Martin","points":0,  "color":"#358C75"},
}

TEAM_STRENGTH_2026 = {
    "Mercedes":10,"Ferrari":9,"McLaren":8,"Red Bull":7,
    "Alpine":6,"Racing Bulls":5,"Haas":4,"Williams":4,
    "Audi":4,"Aston Martin":3,"Cadillac":3,
}
FEATURE_COLS_PRED = [
    "round","avg_pos_last3","team_form_last3","pace_delta",
    "driver_experience","team_strength","cumulative_points","dnf_rate",
]
DRIVER_EXP_2026 = {
    "ANT":2,"RUS":7,"HAM":20,"LEC":7,"NOR":6,"PIA":3,"VER":9,"HAD":1,
    "GAS":10,"COL":2,"LAW":3,"LIN":1,"BEA":2,"OCO":9,"BOR":1,"HUL":12,
    "SAI":10,"ALB":7,"BOT":14,"PER":10,"ALO":24,"STR":7,
}
DNF_RATES_2026 = {
    "ANT":0.10,"RUS":0.06,"HAM":0.05,"LEC":0.08,"NOR":0.07,"PIA":0.08,
    "VER":0.06,"HAD":0.12,"GAS":0.07,"COL":0.13,"LAW":0.10,"LIN":0.14,
    "BEA":0.10,"OCO":0.08,"BOR":0.15,"HUL":0.07,"SAI":0.07,"ALB":0.08,
    "BOT":0.09,"PER":0.09,"ALO":0.08,"STR":0.10,
}

import re as _re

# ── Legacy CIRCUIT_PATHS removed — SVG data now lives in TRACK_DATA[n]["svg_path"]
# ──────────────────────────────────────────────────────────────────────────────
_LEGACY_PLACEHOLDER = {
    "Spa-Francorchamps": "M 220,45 C 242,45 258,60 255,80 C 252,100 235,108 215,110 L 95,198 C 78,214 75,228 84,242 L 90,252 C 96,262 90,272 78,275 L 52,278 C 36,280 28,268 32,252 L 38,75 C 40,55 56,42 76,43 Z",
    "Monaco": "M 158,258 C 170,258 180,248 182,236 L 186,175 C 188,158 200,150 215,153 L 232,156 C 246,159 254,148 250,134 L 240,82 C 236,66 222,60 206,63 L 138,68 C 122,70 112,60 108,46 L 103,30 C 99,18 85,14 72,18 L 46,36 C 32,48 30,64 36,78 L 44,198 C 46,220 56,238 74,245 L 138,260 Z",
    "Silverstone": "M 75,52 L 222,52 C 244,52 256,66 253,88 L 248,115 C 246,130 258,144 258,162 C 258,180 244,192 228,188 L 212,182 C 196,175 180,184 177,198 L 172,234 C 168,252 152,262 134,258 L 72,252 C 50,248 38,232 40,210 L 42,82 C 42,62 57,50 75,52 Z",
    "Monza": "M 58,255 L 245,255 C 258,255 265,242 260,228 L 253,212 C 248,198 255,186 264,182 L 268,138 C 270,120 260,108 246,112 L 230,116 C 216,120 208,108 212,94 L 218,52 C 222,36 210,26 196,30 L 70,32 C 52,34 42,48 44,65 L 50,208 C 50,232 54,252 58,255 Z",
    "Budapest": "M 152,42 C 174,40 188,54 185,74 L 180,96 C 177,112 188,126 205,128 L 226,130 C 242,132 250,148 244,164 L 233,184 C 225,200 230,216 244,226 L 252,236 C 260,248 255,262 240,265 L 78,268 C 60,270 48,254 50,236 L 53,188 C 54,170 42,157 34,144 L 32,104 C 30,85 44,70 62,68 L 122,44 Z",
    "Zandvoort": "M 148,36 C 172,34 188,50 185,72 L 180,98 C 177,115 188,128 204,130 L 222,132 C 240,132 250,148 246,165 L 238,192 C 230,212 216,224 198,220 L 174,215 C 156,210 145,225 145,245 C 145,264 128,275 110,267 C 92,260 86,242 93,224 L 100,190 C 106,168 93,154 76,152 L 58,150 C 40,148 30,133 34,115 L 42,70 C 46,50 64,36 86,35 Z",
    "Albert Park": "M 100,48 L 210,48 C 228,48 240,62 238,80 L 234,120 C 232,138 245,150 258,155 L 265,185 C 268,202 258,215 242,215 L 215,212 C 198,210 186,222 184,240 L 180,258 C 176,272 162,278 148,274 L 72,265 C 54,260 44,244 48,226 L 52,85 C 54,64 70,48 90,48 Z",
    "Shanghai": "M 85,52 L 185,52 C 205,52 218,66 215,86 L 210,125 C 208,145 222,158 238,158 L 255,158 C 270,158 278,172 272,188 L 258,225 C 250,245 230,255 210,250 L 190,245 C 170,240 158,255 156,275 L 155,285 C 154,295 142,300 130,296 C 118,292 112,280 115,268 L 120,238 C 124,218 110,205 92,205 L 70,204 C 50,203 38,188 42,170 L 48,80 C 50,62 66,50 85,52 Z",
    "Suzuka": "M 148,35 C 170,33 184,50 180,70 L 174,95 C 170,115 182,130 198,130 C 214,130 225,118 222,102 L 218,75 C 215,58 228,45 245,48 C 262,51 270,68 265,84 L 255,135 C 248,158 228,168 208,162 L 178,155 C 158,148 145,160 142,180 L 138,215 C 134,238 118,252 98,248 C 78,244 66,226 70,206 L 76,165 C 80,145 68,130 50,128 L 38,100 C 32,80 46,62 65,58 L 120,38 Z",
    "Miami": "M 92,50 L 215,50 C 232,50 243,64 240,82 L 235,110 C 232,128 244,142 258,145 L 268,175 C 272,192 260,205 244,202 L 220,198 C 202,194 192,208 190,228 L 186,252 C 182,268 166,276 150,270 L 68,258 C 50,252 40,234 44,215 L 48,82 C 50,62 68,48 88,50 Z",
    "Montreal": "M 105,45 L 205,45 C 224,45 236,60 232,78 L 226,112 C 222,132 235,146 250,148 L 262,180 C 266,198 254,212 238,210 L 218,206 C 200,202 188,218 185,238 C 182,258 165,268 148,262 L 62,248 C 44,242 34,224 38,205 L 42,80 C 44,60 62,44 82,45 Z",
    "Red Bull Ring": "M 138,55 C 162,50 180,66 176,88 L 170,120 C 166,142 180,158 198,158 L 225,158 C 248,158 260,175 252,198 L 238,232 C 228,255 205,264 182,255 L 88,238 C 64,228 52,204 60,180 L 68,142 C 74,118 58,100 50,82 C 44,65 58,50 75,52 L 115,54 Z",
    "Madrid": "M 98,48 L 208,48 C 228,48 240,64 236,84 L 230,118 C 226,138 240,152 255,155 L 262,185 C 265,205 252,218 236,215 L 212,210 C 194,206 182,220 180,242 L 176,260 C 172,276 156,282 140,276 L 62,262 C 44,256 34,238 38,218 L 42,82 C 44,62 62,46 82,48 Z",
    "Baku": "M 68,262 L 238,262 C 256,262 266,246 260,228 L 252,195 C 246,178 258,164 270,158 L 270,100 C 270,80 255,68 238,72 L 215,76 C 198,80 188,68 190,50 C 192,32 175,22 158,28 L 52,50 C 34,56 26,74 32,92 L 40,225 C 44,248 54,264 68,262 Z",
    "Marina Bay": "M 152,255 C 166,255 177,244 178,230 L 180,188 C 181,172 194,163 210,166 L 230,170 C 246,174 256,162 252,146 L 244,105 C 238,88 224,80 208,84 L 185,88 C 168,92 157,80 154,64 L 150,45 C 147,30 133,24 120,30 L 50,56 C 34,64 28,82 34,98 L 40,188 C 42,215 52,240 70,250 L 130,257 Z",
    "Austin": "M 80,50 C 58,50 44,66 46,88 L 50,125 C 52,145 40,160 35,178 C 30,196 38,215 56,220 L 90,228 C 108,232 115,248 110,265 C 106,278 118,288 132,285 L 225,265 C 245,260 255,242 250,222 L 242,195 C 236,175 248,160 260,155 L 265,120 C 268,100 255,86 238,88 L 215,90 C 198,92 186,78 188,62 C 190,46 175,38 160,44 L 95,50 Z",
    "Mexico City": "M 95,50 L 208,50 C 228,50 240,66 236,86 L 230,120 C 226,140 240,155 256,158 L 264,188 C 268,208 254,222 238,218 L 212,212 C 194,208 182,224 180,245 C 178,265 160,275 142,268 L 60,252 C 42,244 32,225 36,205 L 40,82 C 42,62 60,48 80,50 Z",
    "São Paulo": "M 162,45 C 185,42 200,58 196,80 L 190,108 C 186,128 172,140 155,136 C 138,132 128,118 132,100 C 136,82 125,68 108,66 L 65,68 C 46,70 35,86 38,106 L 42,178 C 44,200 38,218 28,232 C 18,246 22,264 38,270 L 185,275 C 205,278 218,262 215,242 L 210,205 C 206,185 220,170 238,170 L 250,200 C 256,218 248,235 232,238 L 218,242",
    "Las Vegas": "M 70,262 L 240,262 C 258,262 268,246 262,228 L 255,155 C 252,136 264,122 275,115 L 275,82 C 275,64 260,50 242,54 L 218,58 C 200,62 188,50 188,34 C 188,18 170,10 154,18 L 48,48 C 30,56 22,76 28,94 L 35,228 C 38,248 52,264 70,262 Z",
    "Yas Island": "M 88,50 L 215,50 C 234,50 246,66 242,86 L 235,125 C 232,145 246,160 260,162 L 268,192 C 272,212 258,226 242,222 L 215,217 C 196,212 184,228 182,248 C 180,268 162,278 144,270 L 64,255 C 46,248 36,228 40,208 L 44,85 C 46,64 64,48 85,50 Z",
    "Doha": "M 145,40 C 168,38 184,55 180,78 L 174,110 C 170,132 182,148 198,148 L 228,148 C 248,148 260,165 255,185 L 245,220 C 238,242 218,254 196,248 L 92,238 C 70,232 56,212 60,190 L 65,148 C 68,126 55,112 40,108 L 35,78 C 32,58 48,42 68,42 L 120,40 Z",
}  # end _LEGACY_PLACEHOLDER


_TRACK_IMAGE_FILES = {
    1:  "r1_track.png",
    2:  "r2_track.png",
    3:  "r3_track.png",
    4:  "r4_track.png",
    5:  "r5_track.png",
    6:  "r6_track.png",
    7:  "r7_track.png",
    8:  "r8_track.png",
    9:  "r9_track.png",
    10: "r10_track.png",
    11: "r11_track.png",
    12: "r12_track.png",
    13: "r13_track.png",
    14: "r14_track.png",
    15: "r15_track.png",
    16: "r16_track.png",
    17: "r17_track.png",
    18: "r18_track.png",
    19: "r19_track.png",
    20: "r20_track.png",
    21: "r21_track.png",
    22: "r22_track.png",
}

_ASSETS_TRACKS = Path(__file__).resolve().parent / "assets" / "tracks"

def make_track_svg(round_num):
    """Return the /assets/ URL for the track PNG, or None if not available."""
    fname = _TRACK_IMAGE_FILES.get(round_num)
    if fname:
        png_file = _ASSETS_TRACKS / fname
        if png_file.exists():
            return f"/assets/tracks/{fname}"
    return None


def build_track_dna_panel(round_num, race):
    """Track DNA panel: power/downforce sensitivity bars + which teams it favours."""
    from ml_models import get_track_profile
    prof = get_track_profile(race["name"])
    power, downforce = prof["power"], prof["downforce"]

    def bar(label, val, color):
        return html.Div([
            html.Div([
                html.Span(label, style={
                    "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"10px",
                    "fontWeight":"700","letterSpacing":"1.5px","textTransform":"uppercase",
                    "color":"rgba(255,255,255,0.3)",
                }),
                html.Span(f"{val*100:.0f}%", style={
                    "fontFamily":"'JetBrains Mono',monospace","fontSize":"11px",
                    "color":color,"float":"right",
                }),
            ]),
            html.Div(html.Div(style={
                "width":f"{val*100:.0f}%","height":"6px","background":color,
                "borderRadius":"3px","transition":"width 0.4s ease",
            }), style={"background":"rgba(255,255,255,0.06)","borderRadius":"3px",
                       "height":"6px","marginTop":"5px","overflow":"hidden"}),
        ], style={"marginBottom":"12px"})

    if downforce > 0.80:
        favors = "Chassis teams: Ferrari · Red Bull"
        fav_col = C["coral"]
    elif power > 0.80:
        favors = "Engine teams: Mercedes · McLaren · Williams"
        fav_col = C["cyan"]
    else:
        favors = "Balanced — McLaren · Mercedes"
        fav_col = C["gold"]

    ctx = _get_context()
    note = None
    for upd in ctx.get("team_updates", []):
        if (upd.get("from_round") or 99) <= round_num and upd.get("type") == "engine_upgrade":
            note = f"UPGRADE: {upd['team']} +engine from R{upd.get('from_round')}"

    children = [
        shead("Track DNA", "◈"),
        bar("Engine Power", power, C["coral"]),
        bar("Chassis / Downforce", downforce, C["cyan"]),
        html.Div([
            html.Span("FAVORS  ", style={
                "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
                "fontWeight":"700","letterSpacing":"2px","color":"rgba(255,255,255,0.25)",
            }),
            html.Span(favors, style={
                "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"12px",
                "color":fav_col,"fontWeight":"600","letterSpacing":"0.5px",
            }),
        ], style={"marginTop":"8px"}),
    ]
    if note:
        children.append(html.Div(note, style={
            "fontFamily":"'Barlow Condensed',sans-serif",
            "fontSize":"11px","color":C["gold"],"fontWeight":"700","marginTop":"10px",
            "background":"rgba(255,208,128,0.1)","padding":"6px 12px","borderRadius":"8px",
            "display":"inline-block","letterSpacing":"0.5px",
        }))

    return html.Div(children, style={
        "background":C["card"],"border":f"1px solid {C['border']}",
        "borderRadius":"16px","backdropFilter":"blur(20px)",
        "WebkitBackdropFilter":"blur(20px)","padding":"20px 24px","marginTop":"14px",
    })


def find_actual_results(race_name):
    """
    Find actual race results in the demo JSON by race-name match (the ML-tab
    calendar and the JSON use different round numbering, so rounds cannot be
    compared directly).  Returns {code: position} or {} if the race has not
    been run / is not in the data.
    """
    def _core(name):
        return (name.lower().replace(" grand prix", "").replace(" gp", "")
                .replace("(", "").replace(")", "").strip())
    target = _core(race_name)
    for race in DEMO.get("races", []):
        cand = _core(race.get("name", ""))
        if cand and (cand in target or target in cand):
            return {r["code"]: r["position"] for r in race.get("results", [])}
    return {}


def compute_race_prediction(round_num):
    """Run the 18-feature ML ensemble for a race round. Returns (results, error)."""
    if not _load_ml_models():
        return None, "Modelle nicht geladen — bitte zuerst: python ml_models.py --demo"

    race_info = CALENDAR_BY_ROUND.get(round_num)
    if not race_info:
        return None, f"Kein Rennen für Runde {round_num} gefunden"

    from ml_models import build_inference_matrix
    race_name = race_info["name"]
    X, codes, states = build_inference_matrix(
        round_num, race_name, STANDINGS_2026, _get_base_df(), context=_get_context())

    rf_preds  = _rf_model.predict(X)
    xgb_preds = _xgb_model.predict(X)
    ensemble  = 0.6 * rf_preds + 0.4 * xgb_preds

    # Season-level win probabilities from a prior simulation, if available.
    win_probs = _get_sim_summary().get("driver_win_probability", {})

    # Drivers carrying a news-context reliability flag active at this round.
    ctx_flagged = {
        u["driver"] for u in _get_context().get("driver_updates", [])
        if u.get("type") == "reliability_flag"
        and (u.get("from_round") or 99) <= round_num
    }

    order = np.argsort(ensemble)
    results = []
    for rank, idx in enumerate(order, 1):
        code = codes[idx]
        info = STANDINGS_2026[code]
        st = states[code]
        results.append({
            "position":    rank,
            "code":        code,
            "name":        info["name"],
            "team":        info["team"],
            "color":       info["color"],
            "rf_score":    round(float(rf_preds[idx]),  2),
            "xgb_score":   round(float(xgb_preds[idx]), 2),
            "ensemble":    round(float(ensemble[idx]),   2),
            "points":      {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}.get(rank, 0),
            # explainability / state
            "pace_form":   round(float(st["pace_adjusted_form"]), 2),
            "momentum":    round(float(st["momentum_score"]), 2),
            "reliability_flag": int(st["consecutive_dnf_flag"]) or int(code in ctx_flagged),
            "rolling_dnf": round(float(st["rolling_dnf_rate"]), 2),
            "engine":      round(float(st["engine_rating"]), 1),
            "chassis":     round(float(st["chassis_rating"]), 1),
            "track_score": round(float(st["track_performance_score"]), 2),
            "win_prob":    win_probs.get(code),
        })
    return results, None


def compute_shap_prediction(round_num, top_driver_code):
    """TreeSHAP explanation for one driver at a race round (18 features)."""
    if not _load_ml_models():
        return None
    try:
        import shap as _shap
        from ml_models import build_inference_matrix, FEATURE_COLS
        from shap_analysis import FEATURE_LABELS
        race_name = CALENDAR_BY_ROUND.get(round_num, {}).get("name", "")
        X, codes, states = build_inference_matrix(
            round_num, race_name, STANDINGS_2026, _get_base_df(), context=_get_context())
        if top_driver_code not in codes:
            return None
        i = codes.index(top_driver_code)
        explainer = _shap.TreeExplainer(_rf_model)
        svs = explainer.shap_values(X[i:i+1])[0]
        ev = explainer.expected_value
        baseline = float(np.mean(ev) if hasattr(ev, "__len__") else ev)
        return {
            "baseline":   round(baseline, 2),
            "prediction": round(baseline + float(svs.sum()), 2),
            "shap":       {FEATURE_LABELS.get(FEATURE_COLS[j], FEATURE_COLS[j]):
                           round(float(svs[j]), 3) for j in range(len(svs))},
            "shap_values_raw": {FEATURE_COLS[j]: float(svs[j]) for j in range(len(svs))},
            "states": states,
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"SHAP prediction failed: {e}")
        return None


_CARD = {
    "background":C["card"],"border":f"1px solid {C['border']}",
    "borderRadius":"16px","backdropFilter":"blur(20px)",
    "WebkitBackdropFilter":"blur(20px)","padding":"20px 24px","marginBottom":"14px",
}


def build_predictions(rd, d1, d2, year, round_num, **kwargs):
    """Predictions tab — race selector, track info, ML results, SHAP explanation."""
    opts = []
    for r in TRACK_DATA.values():
        tick = "●" if r["completed"] else "○"
        name = r["name"].replace(" Grand Prix","").replace(" GP","")
        circuit_short = r.get("circuit","")[:16]
        opts.append({
            "label": f"{tick} R{r['round']:<2d}  {r['name']:<26s} — {circuit_short}",
            "value": r["round"],
        })
    next_upcoming = next((r["round"] for r in TRACK_DATA.values() if not r["completed"]), 24)

    # Visible banner when no trained models are on disk (graceful degradation).
    model_banner = html.Div()
    if not _load_ml_models():
        model_banner = html.Div(ML_MISSING_BANNER, style={
            "background":"rgba(232,0,45,0.10)","border":"1px solid rgba(232,0,45,0.35)",
            "borderRadius":"8px","padding":"12px 16px","marginBottom":"14px",
            "fontFamily":"'JetBrains Mono',monospace","fontSize":"12px","color":"#ff8888",
        })

    return html.Div([
        model_banner,
        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Div("ML RACE PREDICTIONS", style={
                        "fontFamily":"'Barlow Condensed',sans-serif",
                        "fontSize":"22px","fontWeight":"900","color":C["coral"],
                        "letterSpacing":"2px",
                    }),
                    html.Div("Random Forest + XGBoost · SHAP Explainability · 18 Features", style={
                        "fontFamily":"'JetBrains Mono',monospace",
                        "fontSize":"10px","color":"rgba(255,255,255,0.3)","marginTop":"3px",
                    }),
                ]),
            ], style={"display":"flex","gap":"12px","alignItems":"center"}),
            html.Div([
                html.Span(f"{sum(1 for r in TRACK_DATA.values() if r['completed'])} races completed · {sum(1 for r in TRACK_DATA.values() if not r['completed'])} remaining", style={
                    "fontFamily":"'JetBrains Mono',monospace",
                    "fontSize":"10px","color":"rgba(255,255,255,0.25)",
                }),
            ]),
        ], style={
            **_CARD,
            "display":"flex","justifyContent":"space-between","alignItems":"center",
            "borderLeft":f"3px solid {C['coral']}",
        }),

        # ── Race selector ────────────────────────────────────────────────────
        html.Div([
            shead("Select Race", "▸"),
            dcc.Dropdown(
                id="pred-race-select",
                options=opts,
                value=next_upcoming,
                clearable=False,
                className="dark-dropdown",
                style={"fontFamily":"'JetBrains Mono',monospace","fontSize":"11px"},
            ),
        ], style={**_CARD, "position":"relative", "zIndex":1000}),

        # ── Track info panel (filled by callback) ────────────────────────────
        html.Div(id="pred-track-panel", style={"position":"relative","zIndex":1}),

        # ── Run prediction button (predictions also refresh on race change) ──
        html.Div([
            html.Button(
                "RUN ML PREDICTION",
            id="pred-run-btn", n_clicks=0, style={
                "background":"linear-gradient(135deg, #ffb4a7, #ff7a6b)",
                "border":"none","borderRadius":"8px",
                "color":"#0e0e0e","padding":"11px 32px",
                "fontFamily":"'Barlow Condensed',sans-serif",
                "fontSize":"13px","fontWeight":"700",
                "letterSpacing":"1.5px","textTransform":"uppercase",
                "cursor":"pointer",
                "boxShadow":"0 0 24px rgba(255,180,167,0.25)",
            }),
            html.Span("RF 60% + XGBoost 40%  ·  SHAP TreeExplainer", style={
                "fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"9px","color":"rgba(255,255,255,0.2)","marginLeft":"16px",
            }),
        ], style={"marginBottom":"16px","display":"flex","alignItems":"center"}),

        # ── Results area ─────────────────────────────────────────────────────
        html.Div(id="pred-results-panel"),
    ], style={"maxWidth":"1100px"})


# ── Callback: update track panel when race changes ────────────────────────────
@app.callback(
    Output("pred-track-panel", "children"),
    Input("pred-race-select", "value"),
)
def update_track_panel(round_num):
    if round_num is None:
        round_num = 12
    race = CALENDAR_BY_ROUND.get(round_num, _TRACK_FALLBACK)

    track_img_url = make_track_svg(round_num)

    completed = race.get("completed", False)
    status_label = "COMPLETED" if completed else "UPCOMING"
    status_col   = C["cyan"] if completed else C["coral"]
    status_bg    = "rgba(126,244,244,0.08)" if completed else "rgba(255,180,167,0.08)"

    # Track strength table
    strength_rows = []
    for team, stars in race.get("track_strengths", {}).items():
        team_col = TEAM_COLORS.get(team, "#888")
        strength_rows.append(html.Tr([
            html.Td(html.Span(team, style={
                "color":team_col,"fontWeight":"600","fontSize":"10px",
                "fontFamily":"'Barlow Condensed',sans-serif","letterSpacing":"0.5px",
            })),
            html.Td(stars, style={"color":C["gold"],"fontSize":"12px","letterSpacing":"2px"}),
        ], style={"borderBottom":f"1px solid {C['border']}"}))

    def info_row(label, value, mono=False):
        return html.Div([
            html.Span(label, style={
                "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"10px",
                "letterSpacing":"1.5px","textTransform":"uppercase","color":"rgba(255,255,255,0.3)",
                "minWidth":"90px","display":"inline-block",
            }),
            html.Span(value, style={
                "fontFamily":"'JetBrains Mono',monospace" if mono else "'Barlow Condensed',sans-serif",
                "fontSize":"12px","color":C["white"],
            }),
        ], style={"marginBottom":"7px"})

    return html.Div([
        html.Div([
            html.Div([
                # Left: track image
                html.Div([
                    html.Div(
                        html.Img(
                            src=track_img_url,
                            style={"width":"300px","height":"300px","objectFit":"contain","borderRadius":"8px","background":"#0d0d0d","display":"block"},
                        ) if track_img_url else html.Div("No track image", style={"width":"300px","height":"300px","background":"#0d0d0d","borderRadius":"8px","display":"flex","alignItems":"center","justifyContent":"center","color":"rgba(255,255,255,0.2)","fontSize":"12px"}),
                        style={"textAlign":"center"},
                    ),
                ], style={"minWidth":"300px"}),

                # Middle: Race info
                html.Div([
                    html.Div([
                        html.Span(f"R{race.get('round','')}  ", style={
                            "fontFamily":"'JetBrains Mono',monospace","fontSize":"11px",
                            "color":"rgba(255,255,255,0.3)",
                        }),
                        html.Span(race.get("name",""), style={
                            "fontFamily":"'Barlow Condensed',sans-serif",
                            "fontSize":"18px","fontWeight":"700","color":C["coral"],
                            "letterSpacing":"1px",
                        }),
                    ], style={"marginBottom":"12px"}),
                    # Status badge
                    html.Div(status_label, style={
                        "display":"inline-block","padding":"3px 10px",
                        "border":f"1px solid {status_col}","borderRadius":"20px",
                        "fontSize":"9px","fontWeight":"700","letterSpacing":"2px",
                        "fontFamily":"'Barlow Condensed',sans-serif",
                        "color":status_col,"background":status_bg,"marginBottom":"14px",
                    }),
                    info_row("Circuit", race.get("circuit","—")),
                    info_row("Country", race.get("country","—")),
                    info_row("Date", race.get("date","—"), mono=True),
                    info_row("Length", f"{race.get('length_km','—')} km", mono=True),
                    info_row("Corners", str(race.get("corners","—")), mono=True),
                    info_row("DRS Zones", str(race.get("drs_zones","—")), mono=True),
                    info_row("Lap Record", race.get("lap_record","—"), mono=True),
                    info_row("First GP", str(race.get("first_gp","—")), mono=True),
                    html.Div(
                        [html.Span(c, style={
                            "background":"rgba(255,180,167,0.08)","border":f"1px solid rgba(255,180,167,0.2)",
                            "borderRadius":"12px","padding":"3px 10px","fontSize":"9px",
                            "fontFamily":"'Barlow Condensed',sans-serif","letterSpacing":"0.5px",
                            "color":"rgba(255,180,167,0.6)","display":"inline-block",
                        }) for c in race.get("characteristics", [])],
                        style={"display":"flex","flexWrap":"wrap","gap":"5px","marginTop":"10px"},
                    ),
                ], style={"flex":"1","padding":"0 20px"}),

                # Right: Team strengths
                html.Div([
                    shead("Track Favourites", "★"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("TEAM", style={
                                "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
                                "color":"rgba(255,255,255,0.25)","padding":"4px 8px","letterSpacing":"1.5px",
                                "borderBottom":f"1px solid {C['border']}",
                            }),
                            html.Th("STRENGTH", style={
                                "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
                                "color":"rgba(255,255,255,0.25)","padding":"4px 8px","letterSpacing":"1.5px",
                                "borderBottom":f"1px solid {C['border']}",
                            }),
                        ])),
                        html.Tbody(strength_rows),
                    ], style={"width":"100%","borderCollapse":"collapse"}),
                ], style={"minWidth":"200px"}),
            ], style={"display":"flex","gap":"20px","flexWrap":"wrap","alignItems":"flex-start"}),
        ], style={**_CARD, "marginBottom":"14px"}),

        # ── Track DNA panel (power/downforce sensitivity) ────────────────────
        build_track_dna_panel(round_num, race),
    ])


# ── Callback: run ML prediction ───────────────────────────────────────────────
# The race dropdown is wired as an Input (not State): selecting a different
# Grand Prix re-runs inference and refreshes every chart on this tab without
# needing the button.  The button remains as an explicit re-run trigger.
@app.callback(
    Output("pred-results-panel", "children"),
    [Input("pred-race-select", "value"),
     Input("pred-run-btn", "n_clicks")],
)
def run_prediction(round_num, n_clicks):
    if round_num is None:
        return html.Div()

    try:
        results, err = compute_race_prediction(round_num)
    except Exception as _exc:
        import traceback as _tb
        return html.Div([
            html.Div(f"Fehler in compute_race_prediction: {_exc}", style={
                "color":"#ff4444","padding":"16px","fontFamily":"monospace","fontSize":"12px",
                "whiteSpace":"pre-wrap",
            }),
            html.Div(_tb.format_exc(), style={
                "color":"#aaa","padding":"0 16px 16px","fontFamily":"monospace","fontSize":"10px",
                "whiteSpace":"pre-wrap",
            }),
        ], style={"background":C["card"],"border":"1px solid #ff4444","borderRadius":"8px"})
    race = CALENDAR_BY_ROUND.get(round_num, {})

    if err:
        return html.Div([
            html.Div(ML_MISSING_BANNER, style={"color":C["accent"],"padding":"20px","fontSize":"12px",
                                               "fontFamily":"'JetBrains Mono',monospace"}),
            html.Div(err, style={"color":C["muted"],"padding":"0 20px 20px","fontSize":"11px",
                                 "fontFamily":"monospace"}),
        ], style={"background":C["card"],"border":f"1px solid {C['border']}",
                  "borderRadius":"8px"})

    top10 = results[:10]
    winner = top10[0]

    # SHAP explanation for the predicted winner
    shap_data = compute_shap_prediction(round_num, winner["code"])

    # ── Top 10 predicted grid ────────────────────────────────────────────────
    _TH = {
        "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
        "color":"rgba(255,255,255,0.3)","textTransform":"uppercase","letterSpacing":"2px",
        "padding":"8px 10px","borderBottom":f"1px solid {C['border']}",
    }

    grid_rows = []
    PTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    for r in top10:
        col = r["color"]
        pts = PTS.get(r["position"], 0)
        medal = f"P{r['position']}"
        grid_rows.append(html.Tr([
            html.Td(medal, style={"padding":"7px 10px","fontSize":"13px","textAlign":"center",
                                  "fontFamily":"'JetBrains Mono',monospace"}),
            html.Td(html.Span(r["code"], style={
                "color":col,"fontWeight":"700",
                "fontFamily":"'JetBrains Mono',monospace","fontSize":"12px",
            })),
            html.Td(r["name"], style={
                "color":C["text"],"fontSize":"12px","padding":"7px 10px",
                "fontFamily":"'Barlow Condensed',sans-serif","fontWeight":"600",
            }),
            html.Td(html.Span(r["team"], style={
                "color":col,"fontSize":"10px","opacity":"0.7",
                "fontFamily":"'Barlow Condensed',sans-serif",
            }), style={"padding":"7px 10px"}),
            html.Td(f"+{pts}" if pts else "—", style={
                "color":C["coral"] if pts else C["muted"],
                "fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"11px","fontWeight":"600","textAlign":"right","padding":"7px 10px",
            }),
            html.Td(
                f"{r['win_prob']:.1f}%" if r.get("win_prob") is not None else "—",
                style={"fontFamily":"'JetBrains Mono',monospace","fontSize":"11px",
                       "color":C["cyan"],"padding":"7px 10px","textAlign":"right"},
            ),
            html.Td([
                html.Span(f"Δ{r['pace_form']:.1f}", style={
                    "fontFamily":"'JetBrains Mono',monospace",
                    "fontSize":"10px","color":"rgba(255,255,255,0.35)","marginRight":"6px",
                }),
                (html.Span("DNF RISK", style={
                    "fontSize":"9px","color":C["coral"],"fontWeight":"700",
                    "fontFamily":"'Barlow Condensed',sans-serif","letterSpacing":"0.5px",
                    "background":"rgba(255,180,167,0.12)","padding":"2px 7px",
                    "borderRadius":"8px","border":"1px solid rgba(255,180,167,0.3)",
                }) if r.get("reliability_flag") else html.Span()),
            ], style={"padding":"7px 10px"}),
            html.Td([
                html.Div(style={
                    "height":"5px",
                    "width":f"{max(4, int((22 - r['position']) / 21 * 100))}px",
                    "background":f"linear-gradient(90deg,{C['coral']},{C['cyan']})",
                    "borderRadius":"3px","opacity":"0.6",
                }),
            ], style={"padding":"7px 10px"}),
        ], style={"borderBottom":f"1px solid {C['border']}"}))

    def th(label, right=False):
        s = {**_TH}
        if right:
            s["textAlign"] = "right"
        return html.Th(label, style=s)

    results_table = html.Table([
        html.Thead(html.Tr([
            th(""),th("DRV"),th("Driver"),th("Team"),
            th("PTS",right=True),th("WIN %",right=True),th("Pace / Flags"),th(""),
        ])),
        html.Tbody(grid_rows),
    ], style={"width":"100%","borderCollapse":"collapse","fontSize":"12px"})

    # ── SHAP feature attribution (top 5 features for this race) ─────────────
    shap_section = html.Div()
    if shap_data:
        sorted_shap = sorted(shap_data["shap"].items(),
                             key=lambda x: abs(x[1]), reverse=True)[:5]
        bars = []
        for feat, val in sorted_shap:
            bar_w = int(abs(val) / 6.0 * 180)
            bar_col = C["coral"] if val > 0 else C["cyan"]
            direction = "worsens position" if val > 0 else "improves position"
            bars.append(html.Div([
                html.Div(feat, style={
                    "width":"170px","flexShrink":"0","textAlign":"right","paddingRight":"10px",
                    "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"10px",
                    "color":"rgba(255,255,255,0.35)","letterSpacing":"0.3px",
                }),
                html.Div(style={
                    "width":f"{max(bar_w, 4)}px","height":"10px","background":bar_col,
                    "borderRadius":"2px","opacity":"0.75","flexShrink":"0",
                }),
                html.Span(f" {val:+.3f}  {direction}", style={
                    "fontFamily":"'JetBrains Mono',monospace",
                    "fontSize":"9px","color":bar_col,"marginLeft":"8px",
                }),
            ], style={"display":"flex","alignItems":"center","marginBottom":"7px"}))

        shap_section = html.Div([
            shead(f"Why {winner['code']} wins — SHAP Top-5 Feature Attribution", "◈"),
            html.Div([
                html.Div([
                    html.Span("Baseline  ", style={
                        "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"10px",
                        "color":"rgba(255,255,255,0.3)","letterSpacing":"1px",
                    }),
                    html.Span(f"P{shap_data['baseline']:.1f}", style={
                        "fontFamily":"'JetBrains Mono',monospace","fontSize":"12px","color":C["white"],
                    }),
                    html.Span("  →  Predicted  ", style={
                        "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"10px",
                        "color":"rgba(255,255,255,0.3)","letterSpacing":"1px","marginLeft":"12px",
                    }),
                    html.Span(f"P{shap_data['prediction']:.1f}", style={
                        "fontFamily":"'JetBrains Mono',monospace","fontSize":"15px",
                        "fontWeight":"700","color":winner["color"],
                    }),
                ], style={"marginBottom":"16px"}),
                *bars,
                html.Div([
                    html.Span("■ ", style={"color":C["coral"]}),
                    html.Span("Coral = higher predicted position (worse)   ", style={"marginRight":"12px"}),
                    html.Span("■ ", style={"color":C["cyan"]}),
                    html.Span("Cyan = lower position (better)"),
                ], style={
                    "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
                    "color":"rgba(255,255,255,0.25)","marginTop":"12px",
                    "borderTop":f"1px solid {C['border']}","paddingTop":"10px","letterSpacing":"0.5px",
                }),
            ]),
        ], style={**_CARD, "marginTop":"14px"})

    # ── Natural-language explanation block ────────────────────────────────────
    nl_section = html.Div()
    if shap_data and shap_data.get("states"):
        try:
            from shap_analysis import explain_prediction_natural_language
            from ml_models import get_track_profile
            prof = get_track_profile(race.get("name", ""))
            states = {c: {**s, "team": STANDINGS_2026.get(c, {}).get("team", "")}
                      for c, s in shap_data["states"].items()}
            text = explain_prediction_natural_language(
                shap_data, winner["code"], race.get("name", "this circuit"), prof, states)
            nl_section = html.Div([
                shead("Race Engineer Read", "◎"),
                html.Div(text, style={
                    "fontFamily":"'Barlow Condensed',sans-serif",
                    "fontSize":"14px","color":C["white"],"lineHeight":"1.7","fontWeight":"400",
                    "borderLeft":f"3px solid {C['coral']}","paddingLeft":"16px",
                }),
            ], style={**_CARD, "marginTop":"14px"})
        except Exception as e:
            logging.getLogger(__name__).warning(f"NL explanation failed: {e}")

    # ── Championship impact panel ─────────────────────────────────────────────
    impact_rows = []
    running = {c: STANDINGS_2026[c]["points"] for c in STANDINGS_2026}
    for r in results:
        running[r["code"]] += r["points"]
    top8 = sorted(running.items(), key=lambda x: -x[1])[:8]
    for rank, (code, total) in enumerate(top8, 1):
        info = STANDINGS_2026[code]
        gained = next((rr["points"] for rr in results if rr["code"] == code), 0)
        impact_rows.append(html.Tr([
            html.Td(f"P{rank}", style={
                "padding":"6px 10px","fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"10px","color":"rgba(255,255,255,0.25)",
            }),
            html.Td(html.Span(code, style={
                "color":info["color"],"fontWeight":"700",
                "fontFamily":"'JetBrains Mono',monospace","fontSize":"12px",
            }), style={"padding":"6px 10px"}),
            html.Td(f"{info['points']}", style={
                "padding":"6px 10px","fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"11px","color":"rgba(255,255,255,0.3)","textAlign":"right",
            }),
            html.Td(f"+{gained}" if gained else "—", style={
                "padding":"6px 10px","fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"11px","color":C["coral"],"textAlign":"right",
            }),
            html.Td(f"{total}", style={
                "padding":"6px 10px","fontFamily":"'JetBrains Mono',monospace",
                "fontSize":"13px","fontWeight":"700","color":info["color"],"textAlign":"right",
            }),
        ], style={"borderBottom":f"1px solid {C['border']}"}))

    def imp_th(label, right=False):
        return html.Th(label, style={
            **_TH,
            "textAlign":"right" if right else "left",
            "borderBottom":f"1px solid {C['border']}",
        })

    impact_section = html.Div([
        shead(f"Championship Impact — after R{round_num} if prediction holds", "◈"),
        html.Table([
            html.Thead(html.Tr([
                imp_th(""),imp_th("DRIVER"),
                imp_th("CURRENT",right=True),imp_th("+ THIS RACE",right=True),imp_th("NEW TOTAL",right=True),
            ])),
            html.Tbody(impact_rows),
        ], style={"width":"100%","borderCollapse":"collapse"}),
    ], style={**_CARD, "marginTop":"14px"})

    # ── Race info line ────────────────────────────────────────────────────────
    info_line = html.Div(
        f"Grand Prix: {race.get('name','?')} | Round {round_num} | {len(results)} drivers",
        style={"fontFamily":"'JetBrains Mono',monospace","fontSize":"11px",
               "color":"rgba(255,255,255,0.45)","marginBottom":"12px",
               "background":C["card"],"border":f"1px solid {C['border']}",
               "borderRadius":"8px","padding":"8px 14px"})

    # ── Predicted finishing order bar chart ──────────────────────────────────
    fig_order = go.Figure(go.Bar(
        y=[r["code"] for r in top10][::-1],
        x=[r["ensemble"] for r in top10][::-1],
        orientation="h",
        marker_color=[r["color"] for r in top10][::-1],
        text=[f"P{r['position']}  ({r['ensemble']:.2f})" for r in top10][::-1],
        textposition="outside", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b>  ensemble score %{x:.2f}<extra></extra>",
    ))
    fig_order.update_layout(**pl(height=320,
        title=dict(text="Predicted Finishing Order (lower ensemble score = better)",
                   font=dict(size=10, color=C["muted"])),
        xaxis_title="Ensemble score (RF 60% + XGB 40%)"))
    order_section = card(shead("Predicted Finishing Order", "▦"),
                         dcc.Graph(figure=fig_order, config={"displayModeBar": False}),
                         style={"marginBottom":"14px"})

    # ── Predicted vs actual scatter (only if this race has real results) ─────
    scatter_section = html.Div()
    actual_by_code = find_actual_results(race.get("name", ""))
    if actual_by_code:
        pairs = [(r["code"], r["position"], actual_by_code[r["code"]], r["color"])
                 for r in results if r["code"] in actual_by_code]
        if pairs:
            max_pos = max(max(p[1] for p in pairs), max(p[2] for p in pairs)) + 1
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_pos], y=[0, max_pos], mode="lines", showlegend=False,
                line=dict(color=C["muted"], width=1, dash="dash"),
                hoverinfo="skip"))
            fig_scatter.add_trace(go.Scatter(
                x=[p[2] for p in pairs], y=[p[1] for p in pairs],
                mode="markers+text", text=[p[0] for p in pairs],
                textposition="top center", textfont=dict(size=8, color=C["text2"]),
                marker=dict(size=9, color=[p[3] for p in pairs],
                            line=dict(width=1, color="rgba(255,255,255,0.4)")),
                showlegend=False,
                hovertemplate="<b>%{text}</b>  actual P%{x} / predicted P%{y}<extra></extra>"))
            fig_scatter.update_layout(**pl(height=340,
                title=dict(text="Predicted vs Actual Position (diagonal = perfect prediction)",
                           font=dict(size=10, color=C["muted"])),
                xaxis_title="Actual position", yaxis_title="Predicted position"))
            scatter_section = card(shead("Predicted vs Actual", "◎"),
                                   dcc.Graph(figure=fig_scatter,
                                             config={"displayModeBar": False}),
                                   style={"marginBottom":"14px"})

    winner_color = winner["color"]
    return html.Div([
        # Race context (name | round | driver count)
        info_line,

        # Winner banner — coral gradient
        html.Div([
            html.Div([
                html.Div([
                    html.Div("PREDICTED WINNER", style={
                        "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"9px",
                        "letterSpacing":"3px","color":"rgba(255,255,255,0.4)","marginBottom":"4px",
                    }),
                    html.Div(winner["name"], style={
                        "fontFamily":"'Barlow Condensed',sans-serif",
                        "fontSize":"24px","fontWeight":"900","color":C["white"],"letterSpacing":"1px",
                    }),
                    html.Div(winner["team"], style={
                        "fontFamily":"'Barlow Condensed',sans-serif",
                        "fontSize":"12px","color":"rgba(255,255,255,0.5)","marginTop":"2px",
                    }),
                ]),
            ], style={"display":"flex","alignItems":"center","gap":"16px"}),
            html.Div([
                html.Div("ENSEMBLE", style={
                    "fontFamily":"'Barlow Condensed',sans-serif","fontSize":"8px",
                    "letterSpacing":"2.5px","color":"rgba(255,255,255,0.35)","textAlign":"right",
                }),
                html.Div(f"{winner['ensemble']:.2f}", style={
                    "fontFamily":"'JetBrains Mono',monospace",
                    "fontSize":"28px","fontWeight":"500","color":C["white"],"textAlign":"right",
                }),
                html.Div("RF 60% + XGB 40%", style={
                    "fontFamily":"'JetBrains Mono',monospace","fontSize":"8px",
                    "color":"rgba(255,255,255,0.3)","textAlign":"right",
                }),
            ]),
        ], style={
            "display":"flex","justifyContent":"space-between","alignItems":"center",
            "background":"linear-gradient(135deg, rgba(255,180,167,0.18), rgba(255,122,107,0.10))",
            "border":f"1px solid rgba(255,180,167,0.35)","borderRadius":"16px",
            "padding":"18px 24px","marginBottom":"14px",
            "backdropFilter":"blur(20px)",
        }),

        # Predicted finishing order bar chart
        order_section,

        # Predicted vs actual (only for completed races with data)
        scatter_section,

        # Top 10 table
        html.Div([
            shead("Top 10 Predicted — RF + XGBoost Ensemble", "▸"),
            results_table,
        ], style=_CARD),

        # Natural-language race-engineer read
        nl_section,

        # SHAP explanation
        shap_section,

        # Championship impact
        impact_section,
    ])


# ── Prefetch-Hinweis im Startup ───────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*64)
    print("  F1 RACE ENGINEER DASHBOARD 2026  —  v3")
    print("="*64)
    print(f"  Cache:  {CACHE_DIR}")
    print()
    # Load ML models once at startup (graceful: the ML tab shows a banner
    # instead of crashing when the .pkl files have not been generated yet).
    if _load_ml_models():
        print("  [OK] ML models loaded (models/random_forest_model.pkl, models/xgboost_model.pkl)")
    else:
        print(f"  {ML_MISSING_BANNER}")
    print()
    print("  TIPP: Alle Daten vorab herunterladen für maximale Geschwindigkeit:")
    print("        python f1_prefetch.py")
    print()
    print("  Dashboard: http://localhost:8050")
    print("="*64 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)

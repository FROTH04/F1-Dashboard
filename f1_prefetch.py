"""
F1 Data Prefetcher — lädt alle abgeschlossenen Rennen in den FastF1-Cache.
Einmalig ausführen, dann ist das Dashboard sofort schnell.

Aufruf:  python f1_prefetch.py
         python f1_prefetch.py --year 2025
         python f1_prefetch.py --year 2026 --sessions R Q
"""
import fastf1, os, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.set_log_level('WARNING')

def fetch_session(year, round_num, session_type, event_name):
    try:
        sess = fastf1.get_session(year, round_num, session_type)
        sess.load(laps=True, telemetry=True, weather=True, messages=False)
        laps_count = len(sess.laps) if hasattr(sess, '_laps') else 0
        return (True, f"  [OK]    R{round_num:02d} {event_name[:28]:28s} [{session_type}]  {laps_count} laps")
    except Exception as e:
        return (False, f"  [ERROR] R{round_num:02d} {event_name[:28]:28s} [{session_type}]  {str(e)[:50]}")

def main():
    parser = argparse.ArgumentParser(description='F1 Cache Prefetcher')
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--sessions', nargs='+', default=['R', 'Q'], 
                        choices=['R','Q','FP1','FP2','FP3','S'],
                        help='Welche Sessions laden (default: R Q)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallele Downloads (default: 2, max 3 empfohlen)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  F1 Cache Prefetcher — {args.year}")
    print(f"  Sessions: {', '.join(args.sessions)}")
    print(f"  Cache:    {CACHE_DIR}")
    print(f"{'='*60}\n")

    # Schedule abrufen
    try:
        sched = fastf1.get_event_schedule(args.year, include_testing=False)
    except Exception as e:
        print(f"Fehler beim Schedule-Abruf: {e}")
        sys.exit(1)

    today = pd.Timestamp.now().tz_localize(None)
    completed = []
    for _, ev in sched.iterrows():
        ev_date = pd.Timestamp(ev['EventDate'])
        if ev_date.tzinfo:
            ev_date = ev_date.tz_localize(None)
        if ev_date < today:
            for stype in args.sessions:
                completed.append((int(ev['RoundNumber']), stype, ev['EventName']))

    if not completed:
        print("Keine abgeschlossenen Rennen gefunden.")
        sys.exit(0)

    total = len(completed)
    print(f"  {total} Sessions zum Download ({len(completed)//len(args.sessions)} Rennen × {len(args.sessions)} Sessions)\n")

    ok, fail = 0, 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        futures = {
            pool.submit(fetch_session, args.year, rn, st, name): (rn, st, name)
            for rn, st, name in completed
        }
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if success: ok += 1
            else: fail += 1
            elapsed = time.time() - start
            eta = (elapsed / i) * (total - i)
            print(f"[{i:2d}/{total}]  {msg}")
            if i < total:
                print(f"          Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s", end='\r')

    elapsed = time.time() - start
    print(f"\n\n{'='*60}")
    print(f"  Fertig!  {ok} OK  {fail} Fehler  ({elapsed:.0f}s)")
    print(f"  Cache-Größe: ", end='')
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(CACHE_DIR)
        for f in files
    )
    print(f"{total_size / 1024 / 1024:.0f} MB")
    print(f"{'='*60}\n")
    print("  Dashboard starten: python f1_dashboard.py")
    print("  Alle gecachten Daten laden sofort — keine API-Wartezeit.\n")

if __name__ == '__main__':
    main()

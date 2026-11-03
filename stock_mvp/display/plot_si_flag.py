"""Render the si_regime_filter quarantine layer per ticker.

Panel structure and the crimson eviction-block overlay are taken from
pricing/demo/si_filter.py::render_arithmetic_diagnostic_plots. Panels 1-3 are
the same three curves; panel 4 is retargeted from days-to-cover (which is not
stored) onto the short-interest density the filter actually gates on, with the
si_filter_min_sid / si_filter_max_sid thresholds drawn in.

Reads async_short_interest + async_si_flag. Computes nothing.
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.db import engine

POLICY_PATH = os.path.join(PROJECT_ROOT, "pricing", "simulation_policy_big.json")
SAVE_DIR = os.path.join(PROJECT_ROOT, "pricing", "demo", "artifacts", "si_flag")


def currency_formatter(x, p):
    if x >= 1e12: return f'${x*1e-12:.2f}T'
    if x >= 1e9:  return f'${x*1e-9:.2f}B'
    if x >= 1e6:  return f'${x*1e-6:.2f}M'
    return f'${x:,.0f}'


def load_entity(figi):
    return pd.read_sql(text("""
        SELECT s.stamp, s.ticker, s.unadj_price AS p_nominal, s.unadj_volume AS v,
               s.market_cap, s.pct_short_of_market_cap AS pct,
               f.si_flag, f.raw_trigger
        FROM async_short_interest s
        JOIN async_si_flag f ON f.figi_group = s.figi_group AND f.stamp = s.stamp
        WHERE s.figi_group = :figi
        ORDER BY s.stamp
    """), engine, params={"figi": int(figi)})


def render(figi, ticker_str, policy):
    q = load_entity(figi)
    if q.empty:
        print(f"      -> {ticker_str}: no rows")
        return
    q['t'] = pd.to_datetime(q['stamp'])
    for c in ('p_nominal', 'v', 'market_cap', 'pct'):
        q[c] = pd.to_numeric(q[c], errors='coerce')
    q['si_to_cap_ratio'] = q['pct'] / 100.0
    q['short_interest_market_cap'] = q['si_to_cap_ratio'] * q['market_cap']

    global_min_x, global_max_x = q['t'].min(), q['t'].max()

    # --- contiguous quarantine blocks, exactly as si_filter.py builds them ---
    eviction_blocks = []
    in_block, start_t = False, None
    for _, row in q.iterrows():
        if row['si_flag'] and not in_block:
            start_t, in_block = row['t'], True
        elif not row['si_flag'] and in_block:
            eviction_blocks.append((start_t, row['t']))
            in_block = False
    if in_block:
        eviction_blocks.append((start_t, q['t'].max()))

    def inject_red_highlights(ax, is_primary_axis=True):
        for idx, (start, end) in enumerate(eviction_blocks):
            ax.axvspan(start, end, color='crimson', alpha=0.15,
                       label='MDN Risk Quarantine Layer' if (idx == 0 and is_primary_axis) else "")

    fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
    pct_caught = 100.0 * q['si_flag'].mean()
    fig.suptitle(
        f"SI REGIME FILTER QUARANTINE PROFILE: {ticker_str}\n"
        f"Asset Structural Entity Group ID: {figi}   |   "
        f"{len(q):,} bars, {int(q['si_flag'].sum()):,} quarantined ({pct_caught:.1f}%), "
        f"{int(q['raw_trigger'].sum()):,} raw triggers, {len(eviction_blocks)} episodes",
        fontsize=13, fontweight='bold', color='navy', y=0.995)

    # PANEL 1: raw price & volume
    ax1 = axes[0]
    ax1.set_xlim(global_min_x, global_max_x)
    ax1.set_title("1. Raw Market Transaction Curves & Intraday Trading Volumes", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.plot(q['t'], q['p_nominal'], color='tab:purple', lw=1.4, label='Nominal Price Tape')
    ax1.set_ylabel("Transaction Price ($ USD)", fontweight='bold', color='tab:purple')
    ax1.tick_params(axis='y', labelcolor='tab:purple')
    ax1_vol = ax1.twinx()
    ax1_vol.fill_between(q['t'], q['v'], color='tab:gray', alpha=0.10, label='Intraday Volume')
    ax1_vol.set_ylabel("Bar Volume Count", fontweight='bold', color='tab:gray')
    ax1_vol.tick_params(axis='y', labelcolor='tab:gray')
    ax1_vol.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    inject_red_highlights(ax1)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_vol.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=9)

    # PANEL 2: market capitalization
    ax2 = axes[1]
    ax2.set_title("2. Continuous Asset Valuation: Total Corporate Market Capitalization Space", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_ylabel("Market Cap Value (USD)", fontweight='bold')
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(currency_formatter))
    ax2.plot(q['t'], q['market_cap'], color='darkblue', lw=1.6, label='Market Capitalization (Adjusted Stream Projection)')
    inject_red_highlights(ax2)
    ax2.legend(loc='upper left', fontsize=9)

    # PANEL 3: short interest market cap
    ax3 = axes[2]
    ax3.set_title("3. Microstructure Exposure Mass: Realigned Short Interest Market Capitalization ($ Value)", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.set_ylabel("Realigned Short Cap (USD)", fontweight='bold', color='tab:red')
    ax3.tick_params(axis='y', labelcolor='tab:red')
    ax3.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(currency_formatter))
    ax3.plot(q['t'], q['short_interest_market_cap'], color='tab:red', lw=1.5, label='Short Interest Market Capitalization (Ratiometer Adjusted)')
    inject_red_highlights(ax3)
    ax3.legend(loc='upper left', fontsize=9)

    # PANEL 4: the gated quantity itself
    ax4 = axes[3]
    ax4.set_title("4. Filter Input: Short Interest Density (SID) vs Economic Mass Gates", fontsize=11, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.set_ylabel("Short Capital Intensity (% of Market Cap)", fontweight='bold', color='teal')
    ax4.tick_params(axis='y', labelcolor='teal')
    ax4.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
    ax4.plot(q['t'], q['si_to_cap_ratio'], color='teal', ls='--', lw=1.3, label='Short Interest Density (SID Ratio)')
    ax4.axhline(policy["si_filter_min_sid"], color='dimgray', ls=':', lw=1.5,
                label=f"f4 economic mass floor (si_filter_min_sid = {policy['si_filter_min_sid']})")
    ax4.axhline(policy["si_filter_max_sid"], color='black', ls='-.', lw=1.5,
                label=f"hard breach ceiling (si_filter_max_sid = {policy['si_filter_max_sid']})")
    trig = q[q['raw_trigger'] == 1]
    if not trig.empty:
        ax4.scatter(trig['t'], trig['si_to_cap_ratio'], s=7, color='crimson', zorder=5,
                    label=f'Raw gate breach ({len(trig):,} bars)')
    inject_red_highlights(ax4)
    ax4.legend(loc='upper left', fontsize=9)

    os.makedirs(os.path.join(SAVE_DIR, ticker_str), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    target = os.path.join(SAVE_DIR, ticker_str, f"{ticker_str}_si_regime_quarantine_profile.png")
    fig.savefig(target, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"      -> Saved {target}")


def main():
    policy = json.load(open(POLICY_PATH))
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Every entity in the table, heaviest quarantine first, so the most-caught
    # names land before the clean ones. Ticker is unique per figi_group across
    # this table (286 groups, 286 distinct symbols), so one folder per ticker
    # cannot collide.
    targets = pd.read_sql(text("""
        SELECT figi_group, max(ticker) AS ticker, count(*) AS bars,
               sum(si_flag)::float / count(*) AS caught_share, sum(raw_trigger) AS triggers
        FROM async_si_flag
        GROUP BY figi_group
        ORDER BY caught_share DESC, bars DESC
    """), engine)

    print(f"[*] Rendering {len(targets)} entities into {SAVE_DIR}")
    for i, r in enumerate(targets.itertuples(), start=1):
        print(f"  [{i}/{len(targets)}] {r.ticker}: {r.bars:,} bars, "
              f"caught {100*r.caught_share:.1f}%, {r.triggers:,} triggers", flush=True)
        render(r.figi_group, str(r.ticker), policy)

    engine.dispose()


if __name__ == "__main__":
    main()

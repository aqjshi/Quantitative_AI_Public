import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless file rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from millify import millify
from datetime import datetime
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless file rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_production_timelines(
    df_prod: pd.DataFrame,
    eval_start: datetime, 
    eval_end: datetime, 
    current_start: datetime, 
    production_end:datetime,
    today: datetime,
    global_metadata_lookup: Dict[Any, Dict[str, Any]],
    save_path: str,
    max_series: int = 20, 
    production: bool = False
) -> None:
    """Renders single-panel level trajectories for Nowcasts in batches of `max_series` until finished."""
    id_col = 'series_id'
    all_unique_ids = df_prod[id_col].unique()
    total_series = len(all_unique_ids)

    if total_series == 0:
        print("[!] Empty DataFrame passed to plot_production_timelines. Skipping...")
        return

    base_dir, base_file = os.path.split(save_path)
    file_name, file_ext = os.path.splitext(base_file)

    # -----------------------------------------------------------------
    # BATCHING LOOP: Stride by max_series until all series are plotted
    # -----------------------------------------------------------------
    for batch_idx, start_i in enumerate(range(0, total_series, max_series)):
        batch_ids = all_unique_ids[start_i : start_i + max_series]
        num_series = len(batch_ids)

        fig, axs = plt.subplots(
            nrows=num_series,
            ncols=1,
            figsize=(14, 3.2 * num_series),
            sharex=False
        )
        axs_flat = np.atleast_1d(axs)

        for i, s_id in enumerate(batch_ids):
            ax = axs_flat[i]
            sub = df_prod[df_prod[id_col] == s_id].copy()

            sub = sub.sort_values(by=['date', 'current_start'])
            sub = sub.drop_duplicates(subset=['date'], keep='last')

            meta = global_metadata_lookup[s_id]
            series_name = meta['title']

            if production: 
                sub = sub[sub['date'] > eval_end]
                
                # Drop NaNs before plotting or add marker='o' so isolated quarterly points render!
                sub_orc = sub.dropna(subset=['orc_val'])
                sub_pit = sub.dropna(subset=['pit_val'])
                
                ax.plot(sub_pit['date'], sub_pit['pit_val'], label=f"Oracle wrt to {current_start}", color="#2c3e50", linestyle=":", linewidth=10.0, alpha=0.3, marker="s")
                ax.plot(sub_orc['date'], sub_orc['orc_val'], label=f"Oracle wrt to {today}", color="#2c3e50", linewidth=1.8, alpha=0.9, marker="o", markersize=4)
                ax.set_title(f"{series_name} ({s_id}) — Production Levels [{eval_end}, {production_end}]", fontsize=10, fontweight="bold", pad=4)
                
            else: 
                sub = sub[sub['date'] <= current_start]   
                sub_orc = sub.dropna(subset=['orc_val'])
                sub_pit = sub.dropna(subset=['pit_val'])
                
                ax.plot(sub_pit['date'], sub_pit['pit_val'], label=f"Oracle wrt to {current_start}", color="#2c3e50", linestyle=":", linewidth=10.0, alpha=0.3, marker="s")
                ax.plot(sub_orc['date'], sub_orc['orc_val'], label=f"Oracle wrt to {today}", color="#2c3e50", linewidth=1.8, alpha=0.9, marker="o", markersize=4)
                ax.set_title(f"{series_name} ({s_id}) — Evaluation Levels [{eval_start}, {eval_end}]", fontsize=10, fontweight="bold", pad=4)


            ax.plot(sub['date'], sub['kalman_val'], label="Kalman Model Nowcast", color="#e74c3c", linewidth=1.6, linestyle="--", marker="o", markersize=3.5)
            ax.plot(sub['date'], sub['filled_val'], label="Forward Fill Baseline", color="#0c4b00", linestyle=":", linewidth=4, alpha=0.5)

            
            ax.axvline(x=current_start, color="#2980b9", linestyle="-", linewidth=1.8, alpha=0.85, label=f"Current Start ({current_start})")
            ax.axvline(x=eval_end, color="#7f8c8d", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Eval End ({eval_end})")

            ax.set_ylabel("Level", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            ax.tick_params(axis='x', rotation=25, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

        plt.suptitle(
            f"Raw Level Trajectories ({'Oracle' if production else 'PIT'}) — Batch {batch_idx + 1} ({start_i + 1}-{start_i + num_series} of {total_series})",
            fontsize=13, fontweight="bold", y=1.001
        )
        plt.tight_layout()

   
        batch_save_path = os.path.join(base_dir, f"{file_name}_batch_{batch_idx + 1}{file_ext}")
       

        os.makedirs(os.path.dirname(batch_save_path), exist_ok=True)
        plt.savefig(batch_save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] Exported level trajectory batch {batch_idx + 1} ({num_series} series) to: {batch_save_path}")


def analyze_global_backtest(
    all_eval_dfs: List[pd.DataFrame],
    global_metadata_lookup: Dict[Any, Dict[str, Any]],
    meta_dir: str,
    eval_start:datetime,
    eval_end:datetime,
    current_start:datetime,
    production_end: datetime,
    today:datetime,
    config: Dict[str, Any]
) -> None:
    """Master evaluator function aggregating folds, deduplicating, and exporting plot."""
    master_backtest_df = pd.concat(all_eval_dfs, ignore_index=True)
    master_backtest_df['date'] = pd.to_datetime(master_backtest_df['date'])
    # Filter strictly for out-of-sample prediction dates > current_start
    df_prod = master_backtest_df[master_backtest_df['date'] > eval_end].copy()
    df_eval = master_backtest_df[
        (master_backtest_df['date'] >= eval_start) & 
        (master_backtest_df['date'] <= eval_end)
    ].copy()
    orc_prod_chart_path = os.path.join(meta_dir, "oracle_prod.png")
    first_release_prod_chart_path = os.path.join(meta_dir, "first_release_prod.png")

    plot_production_timelines(
        df_prod=df_prod,
        eval_start=eval_start, 
        eval_end=eval_end,
        current_start=current_start,
        production_end=production_end,
        today=today,
        global_metadata_lookup=global_metadata_lookup,
        save_path=orc_prod_chart_path,
        max_series=20, 
        production=True
    )

    plot_production_timelines(
        df_prod=df_eval,
        eval_start=eval_start, 
        eval_end=eval_end,
        current_start=current_start,
        production_end=production_end,
        today=today,
        global_metadata_lookup=global_metadata_lookup,
        save_path=first_release_prod_chart_path,
        max_series=20, 
        production=False
    )

def plot_predictions_scatterplots(
    eval_df: pd.DataFrame,
    save_path: str
) -> None:
    """Renders a 2x2 grid comparing Model & Baseline predictions vs Oracle and PIT targets."""
    if eval_df.empty:
        print(" [!] Empty evaluation DataFrame passed to scatterplots. Skipping plot...")
        return

    df = eval_df.copy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), sharey=False)
    fig.suptitle('Bitemporal Prediction Alignment Diagnostics (Pre-Production)', fontsize=13, fontweight='bold')

    configs = [
        ('orc_val', 'kalman_val', 'Kalman Model vs Oracle Ground Truth', axs[0, 0], '#2980b9'),
        ('pit_val', 'kalman_val', 'Kalman Model vs PIT First-Release', axs[0, 1], '#27ae60'),
        ('orc_val', 'filled_val', 'Forward Fill Baseline vs Oracle', axs[1, 0], '#7f8c8d'),
        ('pit_val', 'filled_val', 'Forward Fill Baseline vs PIT', axs[1, 1], '#8e44ad')
    ]

    for x_col, y_col, title, ax, color in configs:
        if x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, f"Missing {x_col} or {y_col}", ha='center', va='center', transform=ax.transAxes)
            continue

        clean = df[[x_col, y_col]].dropna()
        x_vals = clean[x_col].values
        y_vals = clean[y_col].values

        if len(x_vals) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.scatter(x_vals, y_vals, alpha=0.5, color=color, edgecolors='none', s=18)

        min_v, max_v = min(x_vals), max(x_vals)
        ax.plot([min_v, max_v], [min_v, max_v], color='#e74c3c', linestyle='--', linewidth=1.5, label='45° Perfect Fit')

        mae = np.mean(np.abs(x_vals - y_vals))
        rmse = np.sqrt(np.mean((x_vals - y_vals)**2))
        corr = np.corrcoef(x_vals, y_vals)[0, 1] if np.std(x_vals) > 1e-8 else 0.0
        
        box_text = f"MAE: {millify(mae, precision=3)}\nRMSE: {millify(rmse, precision=3)}\nr: {corr:+.3f}"
        ax.text(0.05, 0.93, box_text, transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

        ax.set_title(title, fontsize=10, fontweight='semibold')
        ax.set_xlabel(f'{x_col.replace("_val", "").upper()} Target Level', fontsize=8)
        ax.set_ylabel(f'{y_col.replace("_val", "").upper()} Forecast Level', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    # 3. Scatterplot (Evaluated only on historical observations)
    scatter_chart_path = os.path.join(meta_dir, "scatterplot_diagnostics.png")
    plot_predictions_scatterplots(
        eval_df=df_eval,
        save_path=scatter_chart_path
    )

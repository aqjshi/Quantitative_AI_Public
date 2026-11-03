import json
import sys 
# ─────────────────────────────────────────────────────────────
# 2. LONGTABLE GENERATION ENGINE
# ─────────────────────────────────────────────────────────────
def generate_latex_longtable(data: list) -> str:
    """
    Parses evaluation JSON records and outputs a fully formatted,
    multi-page LaTeX longtable string using booktabs formatting.
    """
    lines = []
    
    # Table Header Block
    lines.append(r"\begin{small}")
    lines.append(r"\begin{longtable}{cccc}")
    lines.append(r"\caption{Granular Out-of-Sample Evaluation Metrics Across All Backtest Nodes (2016--2025).}")
    lines.append(r"\label{tab:longtable_smape_results} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Target Date} & \textbf{Observed Universe ($N$)} & \textbf{Model sMAPE (\%)} & \textbf{Baseline sMAPE (\%)} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    
    current_year = None
    
    # Process Row Records
    for record in data:
        date_str = record["target_date"]
        year = date_str.split("-")[0]
        
        # Inject subtle horizontal spacing rule between annual boundaries
        if current_year is not None and year != current_year:
            lines.append(r"\addlinespace")
        current_year = year
        
        universe = record["observed_universe"]
        sem_smape = record["sem_overall_smape"].replace("%", r"\%")
        base_smape = record["baseline_overall_smape"].replace("%", r"\%")
        
        # Highlight top-performing model per row
        sem_val = float(record["sem_overall_smape"].rstrip("%"))
        base_val = float(record["baseline_overall_smape"].rstrip("%"))
        
        if sem_val < base_val:
            sem_cell = f"\\textbf{{{sem_smape}}}"
            base_cell = base_smape
        else:
            sem_cell = sem_smape
            base_cell = f"\\textbf{{{base_smape}}}"
            
        row_str = f"{date_str} & {universe} & {sem_cell} & {base_cell} \\\\"
        lines.append(row_str)
        
    lines.append(r"\end{longtable}")
    lines.append(r"\end{small}")
    
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# 3. RUN & PRINT LATEX OUTPUT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Handle optional command-line JSON file or string input

    if len(sys.argv) > 1:
        raw_arg = sys.argv[1]
        try:
            # Check if arg is a path to a JSON file
            with open(raw_arg, 'r') as f:
                input_data = json.load(f)
        except (FileNotFoundError, OSError):
            # Fallback to parsing direct JSON string
            try:
                input_data = json.loads(raw_arg)
            except json.JSONDecodeError:
                print(f"[!] Warning: Could not parse input argument as JSON/file. Falling back to default data.\n")

    latex_output = generate_latex_longtable(input_data)
    print("\n" + "="*80)
    print("                      GENERATED LATEX LONGTABLE OUTPUT")
    print("="*80 + "\n")
    print(latex_output)
    print("\n" + "="*80 + "\n")
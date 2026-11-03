from datetime import date
import json
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_flat_latex_table(values_json_str, metadata_json_str):
    raw_data = json.loads(values_json_str)
    param_metadata = json.loads(metadata_json_str)
    current_date = date.today().strftime("%Y-%m-%d")
    
    # Flatten nested input values (e.g., pulling variables out of 'model' block)
    flat_data = {}
    for key, val in raw_data.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat_data[sub_key] = sub_val
        else:
            flat_data[key] = val

    output = []
    # 1. longtable is self-centering by default, no \begin{table} or \centering needed!
    output.append(r"\begin{longtable}{c|p{4.2cm}|p{2.5cm}|p{5.5cm}}")
    
    # 2. Set up the Caption and Headers for the FIRST page
    output.append(f"\\caption{{Hyperparameter Configuration and Production Values ({current_date})}}")
    output.append(r"\label{tab:system_configuration} \\")
    output.append(r"\toprule")
    output.append(r"\textbf{Symbol} & \textbf{Code Reference} &  \textbf{Production Value} & \textbf{Description} \\")
    output.append(r"\midrule")
    output.append(r"\endfirsthead") # <--- Tells LaTeX this is the start of the table
    
    # 3. OPTIONAL: Header that appears on subsequent pages if it breaks
    output.append(r"\multicolumn{4}{c}{{\bfseries \tablename\ \thetable{} -- Continued from previous page}} \\")
    output.append(r"\toprule")
    output.append(r"\textbf{Symbol} & \textbf{Code Reference} & \textbf{Production Value} & \textbf{Description} \\")
    output.append(r"\midrule")
    output.append(r"\endhead") # <--- Tells LaTeX to repeat this header on page 2, 3, etc.
    
    # 4. Optional footer text at the bottom of a broken page
    output.append(r"\midrule")
    output.append(r"\multicolumn{4}{r}{{Continued on next page...}} \\")
    output.append(r"\endfoot")
    
    # 5. Final bottom rule at the very end of the entire table
    output.append(r"\hline")
    output.append(r"\endlastfoot")
    
    for key, val in flat_data.items():
        # Read directly from the secondary descriptor JSON schema
        meta = param_metadata.get(key, {})
        
        name = meta.get("name", key.replace('_', ' ').title()).replace('&', '\\&')
        notation = meta.get("notation", r"$-$")
        desc = meta.get("desc", "").replace('&', '\\&')
        unit = meta.get("unit", "")
        
        if isinstance(val, list):
            val_str = ", ".join(str(x) for x in val)
            if len(val_str) > 42: 
                val_str = val_str[:39] + "..."
        elif isinstance(val, bool):
            val_str = "True" if val else "False"
        else:
            val_str = str(val)
            
        val_str += unit
        
        output.append(f"{notation} & {name} &  {val_str} & {desc} \\\\")
        output.append(f"\\hline")
        
    if output[-1] == r"\hline":
        output.pop()
        
    output.append(r"\end{longtable}")
    
    
    return "\n".join(output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing parameters.")
        print("Usage: python json_to_latex.py <values_file.json> <metadata_file.json>")
        sys.exit(1)
        
    values_path = sys.argv[1]
    metadata_path = sys.argv[2]
    
    try:
        with open(values_path, 'r') as val_file:
            values_content = val_file.read()
            
        with open(metadata_path, 'r') as meta_file:
            metadata_content = meta_file.read()
            
        latex_table = generate_flat_latex_table(values_content, metadata_content)
        print(latex_table)
        
    except FileNotFoundError as e:
        print(f"Error: Could not locate file -> {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse valid JSON structure -> {e}")
        sys.exit(1)
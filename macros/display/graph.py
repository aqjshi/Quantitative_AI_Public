from typing import List, Tuple, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Secure headless backend for server execution
import matplotlib.pyplot as plt
import pydot
import matplotlib.colors as mcolors  # Added for hex conversion
from datetime import datetime
import os
import json


import html

def _build_stacked_html_label(node_series_id: str, meta_dict: Optional[dict]) -> str:
    raw_title = meta_dict.get("title") or ""
    category_id = meta_dict.get("category_id") or ""
    freq_short = meta_dict.get("frequency_short") or ""
    raw_id = meta_dict.get("series_id") or str(node_series_id)
    
    source_text = raw_title
 
  
    clean_text = source_text.replace(";", " ").replace(",", " ").replace("-", " ").replace("_", " ").replace("\'", " ").replace("\"", " ")
    tokens = [t.strip() for t in clean_text.split(" ") if t.strip()]
    

    lines = []
    current_line = []
    for token in tokens:
        current_line.append(token)
        if len(current_line) >= 3 or token.endswith(":"):
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))
        
    if len(lines) > 5:
        lines = lines[:4] + ["..."]

    # Escape HTML special characters (&, <, >, ", ') for Graphviz
    escaped_lines = [html.escape(line) for line in lines]
    html_lines = "".join(f"{line}<br/>" for line in escaped_lines)
    
    header = f"<b>{html.escape(str(raw_id))} || {html.escape(str(freq_short))} || cat. {html.escape(str(category_id))}</b>"
    
    return f'<{header}<br/>{html_lines}>'


def save_pc_graph_png(
    edge_list: List[Tuple[Any, ...]],
    labels: List[str],
    path: str,
    metadata_lookup: Optional[Dict[Any, dict]] = None, 
    anchor_date: Optional[Any] = None,
    observation_start: Optional[Any] = None,
    observation_end: Optional[Any] = None
):
    """Renders oriented PC graph and saves a matching adjacency list JSON file."""
    try:
        connected = set()
        for item in edge_list:
            connected.add(item[0])
            connected.add(item[1])
            
        graph = pydot.Dot("pc_graph", graph_type="digraph",
                          rankdir="LR", ratio="auto", fontsize="10")
        
        anchor_str = anchor_date.strftime("%Y-%m-%d") if hasattr(anchor_date, "strftime") else str(anchor_date)
        observation_start_str = observation_start.strftime("%Y-%m-%d") if hasattr(observation_start, "strftime") else str(observation_start)
        observation_end_str = observation_end.strftime("%Y-%m-%d") if hasattr(observation_end, "strftime") else str(observation_end)
        graph.set("label", f"PCMCI DAG Strat. Smpl {observation_start_str} -> {observation_end_str}  \\nAnchor (As-Of Date): {anchor_str}")
        graph.set("labelloc", "t")
        graph.set("labeljust", "c")
        graph.set("fontsize", "14")
        
        graph.set_node_defaults(shape="box", style="rounded,filled", 
                                fillcolor="#f8f9fa", color="#6c757d", fontsize="9")
        graph.set_edge_defaults(fontsize="8")
        
        # Generate lightened palette colors
        tab20_hex_palette = []
        for c in plt.cm.tab20.colors:
            rgba = mcolors.to_rgba(c)
            mix_factor = 0.35 
            lightened_rgba = tuple((val * mix_factor) + (1.0 * (1.0 - mix_factor)) for val in rgba[:3])
            tab20_hex_palette.append(mcolors.to_hex(lightened_rgba))

        json_nodes = {}
        json_adjacency = {str(i): [] for i in sorted(connected)}
            
        for i in sorted(connected):
            node_hash_str = labels[i]
            meta_record = metadata_lookup.get(node_hash_str)
            node_fillcolor = "#f8f9fa"  
       
            category_id = meta_record.get("category_id") or ""
            raw_title = meta_record.get("title") 
            raw_id = meta_record.get("series_id") 
          
            color_idx = int(float(category_id)) % 20
            node_fillcolor = tab20_hex_palette[color_idx]
            
            json_nodes[str(i)] = {
                "node_index": i,
                "series_id": raw_id,
                "series_id_hash": node_hash_str,
                "category_id": category_id,
                "title": raw_title
            }
            
            label_markup = _build_stacked_html_label(node_hash_str, meta_record)
            graph.add_node(pydot.Node(str(i), label=label_markup, fillcolor=node_fillcolor))
            
        for item in edge_list:
            s, t, w = item[0], item[1], item[2]
            lag = item[3] if len(item) == 4 else 0
          
            color = "#1f77b4" if w >= 0 else "#d62728"  
            penwidth = 0.3 + 5 * min(1.0, abs(w))
            edge_label = f"({w:.2f}, {lag})"
            
            graph.add_edge(pydot.Edge(str(s), str(t),
                                      label=edge_label,
                                      color=color,
                                      penwidth=str(penwidth)))
            
            json_adjacency[str(s)].append({
                "target_index": int(t),
                "target_series_id": json_nodes[str(t)]["series_id"],
                "weight": float(w),
                "lag": int(lag)
            })

        anchor_str = anchor_date.strftime("%Y-%m-%d") if hasattr(anchor_date, "strftime") else str(anchor_date)
        pc_graph_path = os.path.join(path, f"pc_graph_{anchor_str}.png")
        graph.write_png(pc_graph_path)
        
        obs_start_str = observation_start.strftime("%Y-%m-%d") if hasattr(observation_start, "strftime") else str(observation_start)
        obs_end_str = observation_end.strftime("%Y-%m-%d") if hasattr(observation_end, "strftime") else str(observation_end)

        json_payload = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "anchor_date_as_of": anchor_str,
                "observation_start": obs_start_str,
                "observation_end": obs_end_str,
                "total_nodes": len(json_nodes),
                "total_edges": len(edge_list)
            },
            "nodes": json_nodes,
            "adjacency_list": json_adjacency
        }
        
        json_output_path = os.path.join(path, f"pc_graph_{anchor_str}.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"   [!] pc_graph render or JSON export failed: {e}")

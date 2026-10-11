"""Interactive knowledge graph: papers connected through the methods, datasets and tasks they share."""
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from pyvis.network import Network

COLORS = {"paper": "#4F46E5", "method": "#10B981", "dataset": "#F59E0B", "task": "#EC4899"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9؀-ۿ]", "", name.lower())


def build_graph(papers: Dict[str, Tuple[str, dict]], only_shared: bool = False, height: int = 600):
    """papers: {paper_id: (title, facts)} where facts has methods / datasets / tasks lists.
    Returns (html, shared) where shared lists entities linked to 2+ papers."""
    entities: Dict[Tuple[str, str], dict] = {}
    links = defaultdict(set)
    for pid, (_, facts) in papers.items():
        for etype, field in (("method", "methods"), ("dataset", "datasets"), ("task", "tasks")):
            for name in facts.get(field, []):
                k = (etype, _key(name))
                if not k[1]:
                    continue
                entities.setdefault(k, {"label": name, "type": etype})
                links[k].add(pid)

    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#1f2937", cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, spring_length=140)
    for pid, (title, _) in papers.items():
        net.add_node(f"P:{pid}", label=title if len(title) <= 40 else title[:37] + "...", title=title,
                     color=COLORS["paper"], shape="box", font={"color": "#ffffff", "size": 16})

    shared = []
    for k, info in entities.items():
        n = len(links[k])
        if only_shared and n < 2:
            continue
        node_id = f"{k[0]}:{k[1]}"
        net.add_node(node_id, label=info["label"], title=f"{info['type']} · used by {n} paper(s)",
                     color=COLORS[info["type"]], size=12 + 8 * n, shape="dot")
        for pid in links[k]:
            net.add_edge(f"P:{pid}", node_id, width=1 + n)
        if n >= 2:
            shared.append({"entity": info["label"], "type": info["type"], "papers": n})

    shared.sort(key=lambda x: -x["papers"])
    return net.generate_html(notebook=False), shared


LEGEND = " ".join(
    f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};margin:0 4px 0 12px"></span>{t.capitalize()}'
    for t, c in COLORS.items()
)

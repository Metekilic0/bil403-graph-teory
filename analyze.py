"""
analyze.py
----------
build_graph.py tarafından üretilen actor_graph.graphml dosyasını okuyup
temel graf analizini yapar:
  1. Temel graf istatistikleri
  2. Derece dağılımı grafiği
  3. Merkezilik analizleri (degree, betweenness, eigenvector, closeness, pagerank)
  4. İki oyuncu arasında en kısa yol örneği
  5. Louvain topluluk tespiti + görselleştirme
  6. Tüm sayısal sonuçları CSV ve okunabilir bir rapor (.txt) olarak dışa aktarır

Daha ileri analizler (dayanıklılık testi, ego ağı, link prediction, çap,
kümelenme katsayısı) advanced_analysis.py dosyasındadır - ayrı çalıştırılır.
"""

import os
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import graph_utils as gu

OUTPUT_DIR = "output"
GRAPH_PATH = os.path.join(OUTPUT_DIR, "actor_graph.graphml")

TOP_N = 10
MAX_NODES_FOR_PLOT = 150

report_lines = []  # tüm çıktıyı biriktirip en sonda dosyaya da yazacağız


def log(text=""):
    """Hem ekrana yazdırır hem rapor dosyası için biriktirir."""
    print(text)
    report_lines.append(text)


def load_graph():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("Önce build_graph.py çalıştırıp graf dosyasını üretmelisin.")
    return nx.read_graphml(GRAPH_PATH)


def name_of(G, node):
    return G.nodes[node].get("name", node)


def print_top(title, scores, G):
    log(f"\n--- {title} (İlk {TOP_N}) ---")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    for node, score in ranked:
        log(f"  {name_of(G, node):30s}  {score:.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    G_full = load_graph()

    log("===== TEMEL GRAF İSTATİSTİKLERİ (TÜM GRAF) =====")
    stats = gu.basic_stats(G_full)
    log(f"Düğüm (oyuncu) sayısı: {stats['n_nodes']}")
    log(f"Kenar (ortak film) sayısı: {stats['n_edges']}")
    log(f"Yoğunluk (density): {stats['density']:.5f}")
    log(f"Bağlantılı bileşen sayısı: {stats['n_components']}")

    G = gu.get_largest_component(G_full)
    if not stats["is_connected"]:
        log(f"Analizler en büyük bileşen üzerinden yapılacak: {G.number_of_nodes()} düğüm")

    # ---- Derece dağılımı ----
    degrees = gu.degree_sequence(G)
    plt.figure(figsize=(7, 5))
    plt.hist(degrees, bins=30, color="#4C72B0", edgecolor="black")
    plt.title("Derece Dağılımı (Degree Distribution)")
    plt.xlabel("Derece (kaç farklı oyuncuyla bağlantılı)")
    plt.ylabel("Oyuncu sayısı")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "degree_distribution.png"), dpi=150)
    plt.close()
    log("\nDerece dağılımı grafiği kaydedildi: output/degree_distribution.png")

    # ---- Merkezilik ----
    log("\n===== MERKEZİLİK ANALİZİ =====")
    centrality = gu.centrality_measures(G)

    print_top("Degree Centrality (en çok ortak oyuncuya sahip olanlar)",
               {n: v["degree"] for n, v in centrality.items()}, G)
    print_top("Betweenness Centrality (ağdaki 'köprü' oyuncular)",
               {n: v["betweenness"] for n, v in centrality.items()}, G)
    print_top("Eigenvector Centrality (önemli kişilerle bağlantılı oyuncular)",
               {n: v["eigenvector"] for n, v in centrality.items()}, G)
    print_top("Closeness Centrality (ağın merkezine ortalama en yakın olanlar)",
               {n: v["closeness"] for n, v in centrality.items()}, G)
    print_top("PageRank (Google'ın algoritmasıyla 'itibar' skoru)",
               {n: v["pagerank"] for n, v in centrality.items()}, G)

    # Tüm merkezilik skorlarını CSV'ye aktar (raporda tablo olarak kullanılabilir)
    rows = []
    for node, vals in centrality.items():
        rows.append({"actor": name_of(G, node), **vals})
    df = pd.DataFrame(rows).sort_values("degree", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "centrality_scores.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"\nTüm merkezilik skorları CSV olarak kaydedildi: {csv_path}")

    # ---- En kısa yol örneği ----
    log("\n===== EN KISA YOL ÖRNEĞİ =====")
    nodes = list(G.nodes())
    random.seed(42)
    a, b = random.sample(nodes, 2)
    path = gu.shortest_path_between(G, a, b)
    names = [name_of(G, n) for n in path]
    log(f"{name_of(G, a)}  -->  {name_of(G, b)}")
    log(f"Yol uzunluğu: {len(path) - 1} adım")
    log(" -> ".join(names))

    # ---- Louvain topluluk tespiti ----
    log("\n===== LOUVAIN TOPLULUK TESPİTİ =====")
    partition, modularity = gu.louvain_communities(G)
    num_communities = len(set(partition.values()))
    log(f"Bulunan topluluk sayısı: {num_communities}")
    log(f"Modularity skoru: {modularity:.4f}  (0.3+ genelde 'anlamlı yapı' sayılır)")

    sizes = {}
    for c in partition.values():
        sizes[c] = sizes.get(c, 0) + 1
    log("\nEn büyük 5 topluluk ve boyutları:")
    for c, size in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
        members = [name_of(G, n) for n, com in partition.items() if com == c][:5]
        log(f"  Topluluk {c}: {size} kişi | örnek üyeler: {', '.join(members)}")

    plot_communities(G, partition)

    # ---- Rapor dosyasını kaydet ----
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nTüm çıktı ayrıca metin raporu olarak kaydedildi: {report_path}")
    log("Tüm analizler tamamlandı. Çıktılar 'output/' klasöründe.")


def plot_communities(G, partition):
    if G.number_of_nodes() > MAX_NODES_FOR_PLOT:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:MAX_NODES_FOR_PLOT]
        sub_nodes = [n for n, _ in top_nodes]
        H = G.subgraph(sub_nodes).copy()
    else:
        H = G

    pos = nx.spring_layout(H, seed=42, k=0.3)
    colors = [partition[n] for n in H.nodes()]

    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.2)
    nx.draw_networkx_nodes(H, pos, node_color=colors, cmap=plt.cm.tab20, node_size=150)

    top_labels = sorted(H.degree, key=lambda x: x[1], reverse=True)[:20]
    labels = {n: name_of(H, n) for n, _ in top_labels}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    plt.title("Oyuncu Ağı - Louvain Topluluklarına Göre Renklendirilmiş")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "communities.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"\nTopluluk görselleştirmesi kaydedildi: {path}")


if __name__ == "__main__":
    main()

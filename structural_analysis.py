"""
structural_analysis.py
------------------------
Ağın FORMAL yapısal özelliklerini test eden ileri analizler:

  1. Küçük-dünya (small-world) testi
     -> Ağ, aynı boyuttaki rastgele graflarla karşılaştırıldığında gerçekten
        "küçük dünya" özelliği taşıyor mu? (sigma > 1 ise evet)
  2. Klik (clique) analizi
     -> En büyük "herkesin herkesle bağlı olduğu" oyuncu grupları hangileri?
        (genelde bir filmin kadrosuna karşılık gelir)
  3. K-core ayrıştırması
     -> Ağın en "sıkı bağlı" çekirdek kısmı hangi oyunculardan oluşuyor?

Önce build_graph.py ve analyze.py çalıştırılmış, output/actor_graph.graphml
üretilmiş olmalı.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt

import graph_utils as gu

OUTPUT_DIR = "output"
GRAPH_PATH = os.path.join(OUTPUT_DIR, "actor_graph.graphml")

report_lines = []


def log(text=""):
    print(text)
    report_lines.append(text)


def name_of(G, node):
    return G.nodes[node].get("name", node)


def section_small_world(G):
    log("\n===== KÜÇÜK-DÜNYA (SMALL-WORLD) TESTİ =====")
    log("Ağ, aynı düğüm/kenar sayısına sahip rastgele graflarla karşılaştırılıyor...")

    try:
        result = gu.small_world_metrics(G, n_random=15, seed=42)
    except RuntimeError as e:
        log(f"Test yapılamadı: {e}")
        return

    log(f"Gerçek ağ  -> kümelenme: {result['C_actual']:.4f}  |  ort. yol uzunluğu: {result['L_actual']:.4f}")
    log(f"Rastgele ağ (ort. {result['n_random_trials']} deneme) -> "
        f"kümelenme: {result['C_random_avg']:.4f}  |  ort. yol uzunluğu: {result['L_random_avg']:.4f}")

    sigma = result["sigma"]
    if sigma is None:
        log("Sigma hesaplanamadı (bölme hatası - rastgele ağların kümelenmesi 0 çıktı).")
        return

    log(f"\nSigma (σ) = {sigma:.3f}")
    if sigma > 1:
        log("YORUM: σ > 1, ağ KÜÇÜK-DÜNYA özelliği taşıyor. Yani oyuncular,")
        log("aynı büyüklükteki rastgele bir ağa göre çok daha 'kümelenmiş'")
        log("gruplar halinde ama yine de birbirine şaşırtıcı derecede az adımda ulaşabiliyor.")
    else:
        log("YORUM: σ <= 1, ağ belirgin bir küçük-dünya özelliği göstermiyor.")
        log("Bu, örneklemin çok parçalı/küçük olmasından kaynaklanıyor olabilir.")


def section_cliques(G):
    log("\n===== KLİK (CLIQUE) ANALİZİ =====")
    result = gu.clique_analysis(G, top_n=5)

    log(f"Toplam maksimal klik sayısı: {result['n_cliques']}")
    log(f"En büyük klik boyutu: {result['largest_clique_size']} kişi")

    log("\nEn büyük 5 klik:")
    for i, clique in enumerate(result["top_cliques"], start=1):
        names = [name_of(G, n) for n in clique]
        log(f"  {i}. ({len(clique)} kişi) {', '.join(names)}")

    # Klik boyutu dağılımı grafiği
    plt.figure(figsize=(7, 5))
    plt.hist(result["size_distribution"],
              bins=range(2, result["largest_clique_size"] + 2),
              color="#55A868", edgecolor="black", align="left")
    plt.title("Klik Boyutu Dağılımı")
    plt.xlabel("Klik boyutu (kişi sayısı)")
    plt.ylabel("Kaç adet klik")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "clique_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"\nKlik boyutu dağılımı grafiği kaydedildi: {path}")
    log("Yorum: Film ağlarında her klik genelde tek bir filmin (veya üst üste")
    log("aynı kadroyla çekilmiş birkaç filmin) oyuncularına karşılık gelir,")
    log("çünkü bir filmin kadrosundaki herkes birbirine otomatik bağlanır.")


def section_k_core(G):
    log("\n===== K-CORE AYRIŞTIRMASI =====")
    result = gu.k_core_analysis(G)

    log(f"Maksimum k-core değeri: {result['max_k']}")
    log(f"Ana çekirdek (main core) boyutu: {result['main_core_size']} oyuncu")
    log("Ana çekirdek üyeleri (ağın en 'sıkı bağlı' grubu):")
    for n in result["main_core_nodes"][:15]:
        log(f"  - {name_of(G, n)}  (k-core: {result['core_numbers'][n]})")

    # K-core numaralarına göre renklendirilmiş graf görselleştirmesi
    pos = nx.spring_layout(G, seed=42, k=0.3)
    core_vals = [result["core_numbers"][n] for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.15)
    nodes_drawn = nx.draw_networkx_nodes(
        G, pos, node_color=core_vals, cmap=plt.cm.viridis, node_size=120
    )
    plt.colorbar(nodes_drawn, label="K-core değeri")

    # Sadece ana çekirdekteki oyuncuları etiketle (okunabilirlik için)
    labels = {n: name_of(G, n) for n in result["main_core_nodes"]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title(f"K-Core Ayrıştırması (koyu renkler = daha sıkı bağlı çekirdek, max k={result['max_k']})")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "k_core.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"\nK-core görselleştirmesi kaydedildi: {path}")
    log("Yorum: Ana çekirdekteki oyuncular, ağın geri kalanına göre çok daha")
    log("yoğun bir şekilde birbirine bağlı - bu genelde birden fazla ortak")
    log("filmde tekrar tekrar bir araya gelen küçük bir 'çekirdek kadro'yu işaret eder.")


def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("Önce build_graph.py ve analyze.py çalıştırılmalı.")

    G_full = nx.read_graphml(GRAPH_PATH)
    G = gu.get_largest_component(G_full)
    log(f"Yapısal analizler en büyük bileşen üzerinden yapılıyor: "
        f"{G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar")

    section_small_world(G)
    section_cliques(G)
    section_k_core(G)

    report_path = os.path.join(OUTPUT_DIR, "structural_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nYapısal analiz raporu kaydedildi: {report_path}")
    log("Tüm yapısal analizler tamamlandı.")


if __name__ == "__main__":
    main()

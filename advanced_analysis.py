"""
advanced_analysis.py
---------------------
analyze.py'daki temel analizlerin üzerine, ağın YAPISINI ve DAYANIKLILIĞINI
inceleyen ileri seviye analizler ekler:

  1. Kümelenme katsayısı (clustering coefficient / transitivity)
     -> "Arkadaşlarım birbirini tanıyor mu?"
  2. Çap ve ortalama en kısa yol uzunluğu
     -> "Ağdaki en uzak iki kişi kaç adım uzakta?"
  3. Derece asortativitesi
     -> "Popüler oyuncular birbirleriyle mi çalışıyor, yoksa az bilinen
        oyuncularla mı?"
  4. Kesme noktaları (articulation points)
     -> "Hangi oyuncu çıkarılırsa ağ parçalanır?"
  5. Ağ dayanıklılık (robustness) testi
     -> "En merkezi oyuncuları sırayla çıkarırsak ağ ne kadar hızlı çöker?
        Bunu rastgele çıkarma ile karşılaştır."
  6. En merkezi oyuncunun ego-ağı (kendisi + doğrudan bağlantıları) görseli
  7. Basit link prediction: henüz birlikte oynamamış ama çok ortak
     bağlantısı olan oyuncu çiftleri

Önce analyze.py çalıştırılmış ve output/actor_graph.graphml üretilmiş olmalı.
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


def section_clustering_and_paths(G):
    log("\n===== KÜMELENME KATSAYISI (CLUSTERING) =====")
    cm = gu.clustering_metrics(G)
    log(f"Ortalama yerel kümelenme katsayısı: {cm['average_clustering']:.4f}")
    log(f"Global kümelenme katsayısı (transitivity): {cm['transitivity']:.4f}")
    log("Yorum: 1'e yakın değer, bir oyuncunun film arkadaşlarının da")
    log("birbirini büyük ihtimalle tanıdığı (aynı kadroda oynadığı) anlamına gelir.")
    log("Film ağlarında bu genelde yüksek çıkar çünkü bir filmin tüm kadrosu")
    log("otomatik olarak birbirine bağlanır (klik/clique oluşur).")

    log("\n===== ÇAP VE ORTALAMA EN KISA YOL =====")
    pm = gu.path_metrics(G)
    log(f"Çap (diameter): {pm['diameter']} adım")
    log(f"Ortalama en kısa yol uzunluğu: {pm['avg_shortest_path_length']:.2f} adım")
    log("Yorum: Küçük bir çap, ağın 'küçük dünya' (small-world) özelliği")
    log("taşıdığının işaretidir - herkes birkaç adımda birbirine ulaşabiliyor demektir.")

    log("\n===== DERECE ASORTATİVİTESİ =====")
    r = gu.assortativity(G)
    log(f"Asortativite katsayısı: {r:.4f}")
    if r > 0.05:
        log("Pozitif: Yüksek dereceli (çok bağlantılı) oyuncular birbirleriyle")
        log("çalışma eğiliminde ('yıldızlar yıldızlarla çalışıyor').")
    elif r < -0.05:
        log("Negatif: Yüksek dereceli oyuncular daha çok az-bağlantılı")
        log("oyuncularla çalışıyor (hub-and-spoke yapı).")
    else:
        log("Sıfıra yakın: Belirgin bir eğilim yok, bağlantılar rastgele gibi dağılmış.")


def section_articulation_points(G):
    log("\n===== KESME NOKTALARI (ARTICULATION POINTS) =====")
    points = gu.find_articulation_points(G)
    log(f"Toplam kesme noktası sayısı: {len(points)}")
    if points:
        log("Bu oyuncular çıkarılırsa ağ birden fazla parçaya bölünür:")
        for p in points[:15]:
            log(f"  - {name_of(G, p)}")
    else:
        log("Ağda kesme noktası yok - yani ağ tek bir oyuncunun çıkarılmasıyla")
        log("parçalanamayacak kadar 'sağlam' bağlı (biconnected'a yakın).")
    return points


def section_robustness(G, centrality):
    log("\n===== AĞ DAYANIKLILIK (ROBUSTNESS) TESTİ =====")
    log("En merkezi (betweenness) oyuncuları sırayla çıkarınca ağın en büyük")
    log("parçası ne kadar küçülüyor? Rastgele çıkarmayla karşılaştırılıyor.")

    max_removals = min(30, G.number_of_nodes() - 2)
    targeted = gu.robustness_simulation(G, centrality, metric="betweenness",
                                         max_removals=max_removals)
    random_fail = gu.random_failure_simulation(G, seed=42, max_removals=max_removals)

    x_t, y_t = zip(*targeted)
    x_r, y_r = zip(*random_fail)

    plt.figure(figsize=(8, 5))
    plt.plot(x_t, y_t, marker="o", label="Hedefli saldırı (en merkezi önce)", color="#C44E52")
    plt.plot(x_r, y_r, marker="o", label="Rastgele arıza", color="#4C72B0")
    plt.xlabel("Çıkarılan düğüm sayısı")
    plt.ylabel("En büyük bağlantılı bileşenin boyutu")
    plt.title("Ağ Dayanıklılık Testi: Hedefli Saldırı vs Rastgele Arıza")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "robustness_test.png")
    plt.savefig(path, dpi=150)
    plt.close()

    log(f"\nGrafik kaydedildi: {path}")
    log("Yorum: Eğer kırmızı çizgi maviden çok daha hızlı düşüyorsa, ağ")
    log("'hedefli saldırılara karşı kırılgan ama rastgele arızalara dayanıklı'")
    log("demektir - gerçek dünya sosyal ağlarında (ve çoğu scale-free ağda)")
    log("tipik olarak görülen bir davranıştır.")


def section_ego_network(G, centrality):
    log("\n===== EN MERKEZİ OYUNCUNUN EGO AĞI =====")
    top_actor = max(centrality.keys(), key=lambda n: centrality[n]["degree"])
    log(f"En yüksek degree centrality'ye sahip oyuncu: {name_of(G, top_actor)}")

    ego = gu.ego_network(G, top_actor, radius=1)
    log(f"Ego ağı boyutu: {ego.number_of_nodes()} düğüm, {ego.number_of_edges()} kenar")

    pos = nx.spring_layout(ego, seed=42)
    colors = ["#C44E52" if n == top_actor else "#4C72B0" for n in ego.nodes()]
    sizes = [400 if n == top_actor else 150 for n in ego.nodes()]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(ego, pos, alpha=0.3)
    nx.draw_networkx_nodes(ego, pos, node_color=colors, node_size=sizes)
    labels = {n: name_of(G, n) for n in ego.nodes()}
    nx.draw_networkx_labels(ego, pos, labels=labels, font_size=7)
    plt.title(f"{name_of(G, top_actor)} - Ego Ağı (1. derece bağlantılar)")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ego_network.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Ego ağı görseli kaydedildi: {path}")


def section_link_prediction(G):
    log("\n===== LINK PREDICTION (BASİT ÖNERİ SİSTEMİ) =====")
    log("Henüz birlikte oynamamış ama çok sayıda ortak film arkadaşı olan")
    log("oyuncu çiftleri (gelecekte birlikte çalışması 'beklenebilecek' isimler):")

    predictions = gu.common_neighbor_link_predictions(G, top_n=10)
    if not predictions:
        log("Bu ağda ortak komşusu olan bağlantısız çift bulunamadı (ağ çok küçük/seyrek olabilir).")
        return

    for u, v, common in predictions:
        log(f"  {name_of(G, u):25s} <-> {name_of(G, v):25s}   ortak bağlantı: {common}")


def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("Önce build_graph.py ve analyze.py çalıştırılmalı.")

    G_full = nx.read_graphml(GRAPH_PATH)
    G = gu.get_largest_component(G_full)
    log(f"İleri analizler en büyük bileşen üzerinden yapılıyor: "
        f"{G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar")

    centrality = gu.centrality_measures(G)

    section_clustering_and_paths(G)
    section_articulation_points(G)
    section_robustness(G, centrality)
    section_ego_network(G, centrality)
    section_link_prediction(G)

    report_path = os.path.join(OUTPUT_DIR, "advanced_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nİleri analiz raporu kaydedildi: {report_path}")
    log("Tüm ileri analizler tamamlandı.")


if __name__ == "__main__":
    main()

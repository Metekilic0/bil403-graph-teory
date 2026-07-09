"""
bipartite_analysis.py
-----------------------
Bu proje bugüne kadar doğrudan OYUNCU-OYUNCU ağıyla çalıştı (build_graph.py
zaten arka planda bir "bipartite projection" işlemi yapıyordu, ama bunu
açıkça göstermedi). Bu script, ara adımı görünür kılar:

  1. Ham veriden Oyuncu-Film İKİ PARÇALI (bipartite) grafı kurar
     (iki farklı düğüm tipi: oyuncular ve filmler)
  2. Bu grafı oyuncu-oyuncu ağına PROJEKTE eder (build_graph.py'daki
     sonuçla tutarlı olduğunu doğrular - bkz. tests/test_graph_utils.py)
  3. Aynı grafı FİLM-FİLM ağına da projekte eder (iki film ortak oyuncu
     paylaşıyorsa bağlanır) - bu şimdiye kadar hiç bakmadığımız bir açı
  4. Küçük bir alt kümeyi iki renkli (oyuncu/film) bir görselle gösterir,
     böylece "bipartite" kavramı raporda somut bir örnekle açıklanabilir

Kullanım:
    python bipartite_analysis.py --genre Thriller --min-year 2005 --max-year 2024 --max-movies 60
"""

import argparse
import os

import networkx as nx
import matplotlib.pyplot as plt

import build_graph as bg
import graph_utils as gu

OUTPUT_DIR = "output"


def parse_args():
    p = argparse.ArgumentParser(description="Oyuncu-Film bipartite graf analizi")
    p.add_argument("--genre", type=str, default="Thriller")
    p.add_argument("--min-year", type=int, default=2005)
    p.add_argument("--max-year", type=int, default=2024)
    p.add_argument("--max-movies", type=int, default=60,
                    help="Bipartite görselleştirme okunaklı kalsın diye küçük tutulur")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def name_of(B, node):
    return B.nodes[node].get("name", node)


def plot_bipartite_sample(B, actor_nodes, movie_nodes, movies_df, max_movies_shown=6):
    """Okunabilirlik için sadece birkaç filmi ve oyuncularını gösteren küçük bir örnek çizer."""
    sample_movies = movie_nodes[:max_movies_shown]
    sample_actors = [n for n in actor_nodes if any(B.has_edge(n, m) for m in sample_movies)]
    H = B.subgraph(sample_movies + sample_actors).copy()

    pos = nx.bipartite_layout(H, sample_movies)

    movie_title_map = dict(zip(movies_df["tconst"], movies_df["primaryTitle"]))

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(H, pos, alpha=0.4)
    nx.draw_networkx_nodes(H, pos, nodelist=sample_movies, node_color="#C44E52",
                             node_shape="s", node_size=500, label="Filmler")
    nx.draw_networkx_nodes(H, pos, nodelist=sample_actors, node_color="#4C72B0",
                             node_size=250, label="Oyuncular")

    movie_labels = {m: movie_title_map.get(m, m) for m in sample_movies}
    actor_labels = {a: name_of(H, a) for a in sample_actors}
    nx.draw_networkx_labels(H, pos, labels={**movie_labels, **actor_labels}, font_size=7)

    plt.title("Oyuncu-Film İki Parçalı (Bipartite) Graf Örneği\n(kareler = filmler, daireler = oyuncular)")
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bipartite_sample.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nBipartite örnek görseli kaydedildi: {path}")


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Veri okunuyor ve filtreleniyor...")
    movies = bg.load_filtered_movies(args.genre, args.min_year, args.max_year,
                                       args.max_movies, seed=args.seed)
    if movies.empty:
        raise ValueError("Seçilen tür/yıl aralığında film bulunamadı.")

    cast = bg.load_cast_for_movies(movies["tconst"].tolist())
    if cast.empty:
        raise ValueError("Bu filmler için oyuncu verisi bulunamadı.")

    name_map = bg.load_actor_names(cast["nconst"].unique().tolist())

    print("Bipartite graf (Oyuncu-Film) kuruluyor...")
    B, actor_nodes, movie_nodes = gu.build_bipartite_graph(cast, name_map)
    print(f"Bipartite graf: {len(actor_nodes)} oyuncu düğümü, {len(movie_nodes)} film düğümü, "
          f"{B.number_of_edges()} kenar")

    print("\n===== BİPARTİTE GRAF İSTATİSTİKLERİ =====")
    print(f"Yoğunluk: {nx.density(B):.5f}")
    actor_degrees = [d for n, d in B.degree(actor_nodes)]
    movie_degrees = [d for n, d in B.degree(movie_nodes)]
    print(f"Ortalama oyuncu derecesi (kaç filmde oynadı): {sum(actor_degrees)/len(actor_degrees):.2f}")
    print(f"Ortalama film derecesi (kaç oyuncusu var): {sum(movie_degrees)/len(movie_degrees):.2f}")

    # ---- Projeksiyon 1: Oyuncu-Oyuncu (build_graph.py'daki ile tutarlılık kontrolü) ----
    print("\n===== PROJEKSİYON 1: OYUNCU-OYUNCU AĞI =====")
    actor_graph_from_bipartite = gu.project_bipartite(B, actor_nodes)
    direct_actor_graph = bg.build_graph(cast, name_map)

    edges_from_bipartite = {frozenset(e) for e in actor_graph_from_bipartite.edges()}
    edges_direct = {frozenset(e) for e in direct_actor_graph.edges()}
    consistent = edges_from_bipartite == edges_direct

    print(f"Bipartite projeksiyondan gelen kenar sayısı: {len(edges_from_bipartite)}")
    print(f"build_graph.py'ın doğrudan ürettiği kenar sayısı: {len(edges_direct)}")
    print(f"İki yöntem birbiriyle tutarlı mı? {'EVET' if consistent else 'HAYIR (kontrol edilmeli!)'}")

    # ---- Projeksiyon 2: Film-Film (şimdiye kadar bakılmayan yeni açı) ----
    print("\n===== PROJEKSİYON 2: FİLM-FİLM AĞI (ortak oyuncu paylaşan filmler) =====")
    movie_graph = gu.project_bipartite(B, movie_nodes)
    print(f"Film-film ağı: {movie_graph.number_of_nodes()} düğüm, {movie_graph.number_of_edges()} kenar")

    movie_title_map = dict(zip(movies["tconst"], movies["primaryTitle"]))
    if movie_graph.number_of_edges() > 0:
        top_edge = max(movie_graph.edges(data=True), key=lambda e: e[2].get("weight", 0))
        t1 = movie_title_map.get(top_edge[0], top_edge[0])
        t2 = movie_title_map.get(top_edge[1], top_edge[1])
        print(f"En çok ortak oyuncuyu paylaşan film çifti: '{t1}' <-> '{t2}' "
              f"({top_edge[2].get('weight')} ortak oyuncu)")
    else:
        print("Hiçbir film çifti ortak oyuncu paylaşmıyor (örneklem çok küçük/dağınık olabilir).")

    # ---- Görselleştirme ----
    plot_bipartite_sample(B, actor_nodes, movie_nodes, movies)

    print("\nBipartite analiz tamamlandı.")


if __name__ == "__main__":
    main()

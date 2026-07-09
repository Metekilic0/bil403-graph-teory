"""
compare_genres.py
-------------------
Birden fazla film türü için (varsayılan: Thriller, Comedy, Horror) ayrı ayrı
oyuncu ağı kurar ve temel ağ istatistiklerini karşılaştırmalı bir tabloda
ve grafiklerde sunar.

Soru: "Farklı film türlerinin oyuncu ağları yapısal olarak birbirinden
farklı mı? Örneğin komedi filmlerinde oyuncular arası kümelenme daha mı
yüksek, yoksa gerilim filmlerinde mi?"

Kullanım:
    python compare_genres.py --genres Thriller Comedy Horror --min-year 2005 --max-year 2024 --max-movies 150

Not: Her tür için ayrı bir IMDb okuma/filtreleme işlemi yapılır, bu yüzden
tek bir türe göre build_graph.py çalıştırmaktan daha uzun sürer.
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

import build_graph as bg
import graph_utils as gu

OUTPUT_DIR = "output"


def parse_args():
    p = argparse.ArgumentParser(description="Farklı film türlerinin oyuncu ağlarını karşılaştır")
    p.add_argument("--genres", type=str, nargs="+",
                    default=["Thriller", "Comedy", "Horror"],
                    help="Karşılaştırılacak IMDb tür etiketleri (boşlukla ayrılmış)")
    p.add_argument("--min-year", type=int, default=2005)
    p.add_argument("--max-year", type=int, default=2024)
    p.add_argument("--max-movies", type=int, default=150,
                    help="Her tür için kullanılacak maksimum film sayısı")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def analyze_one_genre(genre, min_year, max_year, max_movies, seed):
    print(f"\n--- '{genre}' türü işleniyor ---")
    movies = bg.load_filtered_movies(genre, min_year, max_year, max_movies, seed=seed)
    if movies.empty:
        print(f"UYARI: '{genre}' için film bulunamadı, atlanıyor.")
        return None

    cast = bg.load_cast_for_movies(movies["tconst"].tolist())
    if cast.empty:
        print(f"UYARI: '{genre}' için oyuncu verisi bulunamadı, atlanıyor.")
        return None

    name_map = bg.load_actor_names(cast["nconst"].unique().tolist())
    G_full = bg.build_graph(cast, name_map)
    G = gu.get_largest_component(G_full)

    stats = gu.basic_stats(G_full)
    clustering = gu.clustering_metrics(G)
    try:
        path_m = gu.path_metrics(G)
    except ValueError:
        path_m = {"diameter": None, "avg_shortest_path_length": None}

    return {
        "genre": genre,
        "n_movies_used": len(movies),
        "n_actors_total": stats["n_nodes"],
        "n_edges_total": stats["n_edges"],
        "density": stats["density"],
        "n_components": stats["n_components"],
        "largest_component_size": G.number_of_nodes(),
        "avg_clustering": clustering["average_clustering"],
        "transitivity": clustering["transitivity"],
        "diameter": path_m["diameter"],
        "avg_shortest_path_length": path_m["avg_shortest_path_length"],
    }


def plot_comparison(df):
    metrics_to_plot = [
        ("density", "Yoğunluk (Density)"),
        ("avg_clustering", "Ortalama Kümelenme Katsayısı"),
        ("largest_component_size", "En Büyük Bileşen Boyutu"),
        ("avg_shortest_path_length", "Ortalama En Kısa Yol Uzunluğu"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, metrics_to_plot):
        ax.bar(df["genre"], df[col], color="#4C72B0", edgecolor="black")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "genre_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nKarşılaştırma grafiği kaydedildi: {path}")


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for genre in args.genres:
        result = analyze_one_genre(genre, args.min_year, args.max_year,
                                     args.max_movies, args.seed)
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("Hiçbir tür için analiz yapılamadı.")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "genre_comparison.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n===== TÜR KARŞILAŞTIRMA TABLOSU =====")
    print(df.to_string(index=False))
    print(f"\nTablo CSV olarak kaydedildi: {csv_path}")

    plot_comparison(df)
    print("\nTür karşılaştırması tamamlandı.")


if __name__ == "__main__":
    main()

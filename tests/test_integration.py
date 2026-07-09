"""
test_integration.py
--------------------
Gerçek IMDb verisine ihtiyaç duymadan, sahte (ama gerçek IMDb formatında)
küçük .tsv.gz dosyaları oluşturup TÜM pipeline'ı (build_graph -> analyze
fonksiyonları -> compare_genres -> bipartite) uçtan uca test eder.

Bu, tests/test_build_graph.py ve tests/test_graph_utils.py'daki birim
testlerinin üstüne, parçaların BİRLİKTE de doğru çalıştığını garanti eder.

Çalıştırmak için proje kök klasöründe:
    pytest tests/test_integration.py -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_graph as bg
import graph_utils as gu
import compare_genres as cg


@pytest.fixture
def fake_imdb_data(tmp_path, monkeypatch):
    """
    tmp_path içinde sahte data/ klasörü oluşturur ve çalışma dizinini oraya
    taşır, böylece build_graph.py'ın fonksiyonları normal şekilde bu
    dosyaları okuyabilir.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")

    basics = pd.DataFrame({
        "tconst": ["tt001", "tt002", "tt003", "tt004", "tt005", "tt006"],
        "titleType": ["movie"] * 6,
        "primaryTitle": ["T1", "T2", "T3", "C1", "C2", "C3"],
        "startYear": ["2015", "2016", "2017", "2015", "2018", "2019"],
        "genres": ["Thriller,Drama", "Thriller", "Thriller,Mystery",
                   "Comedy", "Comedy,Romance", "Comedy"],
    })
    basics.to_csv("data/title.basics.tsv.gz", sep="\t", index=False,
                   compression="gzip", na_rep="\\N")

    cast_map = {
        "tt001": ["nm1", "nm2", "nm3"],
        "tt002": ["nm2", "nm3", "nm4"],
        "tt003": ["nm1", "nm4"],
        "tt004": ["nm5", "nm6"],
        "tt005": ["nm6", "nm7", "nm5"],
        "tt006": ["nm7", "nm8"],
    }
    rows = [{"tconst": t, "nconst": a, "category": "actor"}
            for t, actors in cast_map.items() for a in actors]
    pd.DataFrame(rows).to_csv("data/title.principals.tsv.gz", sep="\t",
                                index=False, compression="gzip", na_rep="\\N")

    names = pd.DataFrame({
        "nconst": [f"nm{i}" for i in range(1, 9)],
        "primaryName": [f"Actor {i}" for i in range(1, 9)],
    })
    names.to_csv("data/name.basics.tsv.gz", sep="\t", index=False,
                  compression="gzip", na_rep="\\N")

    return tmp_path


def test_full_pipeline_produces_valid_graph(fake_imdb_data):
    """build_graph.py'ın tüm okuma/filtreleme/graf kurma adımları uçtan uca çalışmalı."""
    movies = bg.load_filtered_movies("Thriller", 2010, 2020, max_movies=10, seed=1)
    assert len(movies) == 3  # tt001, tt002, tt003

    cast = bg.load_cast_for_movies(movies["tconst"].tolist())
    assert not cast.empty

    name_map = bg.load_actor_names(cast["nconst"].unique().tolist())
    assert name_map["nm1"] == "Actor 1"

    G = bg.build_graph(cast, name_map)
    assert G.number_of_nodes() == 4  # nm1, nm2, nm3, nm4
    assert G.nodes["nm1"]["name"] == "Actor 1"


def test_full_pipeline_stats_are_consistent(fake_imdb_data):
    """Pipeline sonundaki graf üzerinde graph_utils fonksiyonları hatasız çalışmalı."""
    movies = bg.load_filtered_movies("Comedy", 2010, 2020, max_movies=10, seed=1)
    cast = bg.load_cast_for_movies(movies["tconst"].tolist())
    name_map = bg.load_actor_names(cast["nconst"].unique().tolist())
    G = bg.build_graph(cast, name_map)

    stats = gu.basic_stats(G)
    assert stats["n_nodes"] == 4  # nm5, nm6, nm7, nm8
    assert stats["n_edges"] > 0


def test_compare_genres_end_to_end(fake_imdb_data):
    """compare_genres.py'ın tek bir tür için analiz fonksiyonu hatasız sonuç üretmeli."""
    result = cg.analyze_one_genre("Thriller", 2010, 2020, max_movies=10, seed=1)
    assert result is not None
    assert result["genre"] == "Thriller"
    assert result["n_actors_total"] == 4
    assert 0 <= result["density"] <= 1


def test_compare_genres_handles_missing_genre_gracefully(fake_imdb_data):
    """Var olmayan bir tür için None dönmeli, hata fırlatmamalı."""
    result = cg.analyze_one_genre("NonExistentGenreXYZ", 2010, 2020, max_movies=10, seed=1)
    assert result is None


def test_bipartite_projection_matches_pipeline_output(fake_imdb_data):
    """
    Bipartite projeksiyon ile üretilen oyuncu grafı, build_graph.py'ın
    doğrudan ürettiği grafla TAM olarak aynı kenarlara sahip olmalı
    (bipartite_analysis.py'daki tutarlılık kontrolünün birim test karşılığı).
    """
    movies = bg.load_filtered_movies("Thriller", 2010, 2020, max_movies=10, seed=1)
    cast = bg.load_cast_for_movies(movies["tconst"].tolist())
    name_map = bg.load_actor_names(cast["nconst"].unique().tolist())

    direct_graph = bg.build_graph(cast, name_map)
    B, actor_nodes, movie_nodes = gu.build_bipartite_graph(cast, name_map)
    projected_graph = gu.project_bipartite(B, actor_nodes)

    direct_edges = {frozenset(e) for e in direct_graph.edges()}
    projected_edges = {frozenset(e) for e in projected_graph.edges()}
    assert direct_edges == projected_edges

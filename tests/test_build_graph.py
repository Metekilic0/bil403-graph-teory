"""
test_build_graph.py
--------------------
build_graph.py içindeki veri işleme fonksiyonlarını test eder. Gerçek IMDb
dosyalarına ihtiyaç duymaz - küçük, sahte (fake) DataFrame'ler kullanır.

Çalıştırmak için proje kök klasöründe:
    pytest tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

# tests/ klasöründen proje köküne erişim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_graph as bg


# ---------- resolve_data_file ----------

def test_resolve_data_file_prefers_gz(tmp_path, monkeypatch):
    """Hem .gz hem .tsv varsa .gz tercih edilmeli (daha az yer kaplar)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    open(os.path.join("data", "title.basics.tsv.gz"), "w").close()
    open(os.path.join("data", "title.basics.tsv"), "w").close()

    path, compression = bg.resolve_data_file("title.basics.tsv")
    assert path.endswith(".gz")
    assert compression == "gzip"


def test_resolve_data_file_falls_back_to_plain_tsv(tmp_path, monkeypatch):
    """Sadece düz .tsv varsa (kullanıcı zaten çıkarmışsa) onu bulmalı."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    open(os.path.join("data", "name.basics.tsv"), "w").close()

    path, compression = bg.resolve_data_file("name.basics.tsv")
    assert path.endswith("name.basics.tsv")
    assert compression is None


def test_resolve_data_file_raises_when_missing(tmp_path, monkeypatch):
    """Dosya hiç yoksa anlamlı bir hata vermeli."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")

    with pytest.raises(FileNotFoundError):
        bg.resolve_data_file("title.principals.tsv")


def test_resolve_data_file_optional_returns_none_when_missing(tmp_path, monkeypatch):
    """required=False ile çağrıldığında, dosya yoksa hata değil None döner."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")

    path, compression = bg.resolve_data_file("title.ratings.tsv", required=False)
    assert path is None
    assert compression is None


# ---------- load_filtered_movies (popülerlik filtresi ve fallback) ----------

def _write_basics_and_ratings(tmp_path, monkeypatch, n_movies=10, with_ratings=True):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")

    basics = pd.DataFrame({
        "tconst": [f"tt{i:03d}" for i in range(1, n_movies + 1)],
        "titleType": ["movie"] * n_movies,
        "primaryTitle": [f"Movie{i}" for i in range(1, n_movies + 1)],
        "startYear": ["2015"] * n_movies,
        "genres": ["Thriller"] * n_movies,
    })
    basics.to_csv("data/title.basics.tsv.gz", sep="\t", index=False,
                   compression="gzip", na_rep="\\N")

    if with_ratings:
        # numVotes'u kasıtlı olarak karışık sırada veriyoruz
        votes = [100, 5000, 200, 9000, 50, 7000, 10, 3000, 1, 8000][:n_movies]
        ratings = pd.DataFrame({
            "tconst": [f"tt{i:03d}" for i in range(1, n_movies + 1)],
            "numVotes": votes,
        })
        ratings.to_csv("data/title.ratings.tsv.gz", sep="\t", index=False,
                        compression="gzip", na_rep="\\N")


def test_load_filtered_movies_picks_most_popular_when_ratings_available(tmp_path, monkeypatch):
    """
    title.ratings.tsv.gz mevcutsa, max_movies'i aşan durumda RASTGELE değil,
    en çok oy alan (en popüler) filmler seçilmelidir.
    """
    _write_basics_and_ratings(tmp_path, monkeypatch, n_movies=10, with_ratings=True)

    movies = bg.load_filtered_movies("Thriller", 2010, 2020, max_movies=3)

    assert len(movies) == 3
    # En yüksek oylar: tt004 (9000), tt010 (8000), tt006 (7000)
    assert set(movies["tconst"]) == {"tt004", "tt010", "tt006"}


def test_load_filtered_movies_falls_back_to_random_without_ratings(tmp_path, monkeypatch):
    """title.ratings.tsv.gz yoksa hata vermemeli, rastgele örneklemeye düşmelidir."""
    _write_basics_and_ratings(tmp_path, monkeypatch, n_movies=10, with_ratings=False)

    movies = bg.load_filtered_movies("Thriller", 2010, 2020, max_movies=3, seed=1)

    assert len(movies) == 3
    assert "numVotes" not in movies.columns


def test_load_filtered_movies_min_votes_filters_unpopular_movies(tmp_path, monkeypatch):
    """--min-votes verildiğinde, bu eşiğin altındaki filmler tamamen elenmelidir."""
    _write_basics_and_ratings(tmp_path, monkeypatch, n_movies=10, with_ratings=True)

    movies = bg.load_filtered_movies("Thriller", 2010, 2020, max_movies=10, min_votes=3000)

    # 3000+ oy alanlar: tt002(5000), tt004(9000), tt006(7000), tt008(3000), tt010(8000)
    assert len(movies) == 5
    assert all(v >= 3000 for v in movies["numVotes"])


# ---------- build_graph (oyuncu-oyuncu grafı oluşturma) ----------

def test_build_graph_creates_edges_for_costars():
    """Aynı filmde oynayan iki oyuncu arasında bir kenar oluşmalı."""
    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001"],
        "nconst": ["nm001", "nm002"],
        "category": ["actor", "actress"],
    })
    name_map = {"nm001": "Oyuncu Bir", "nm002": "Oyuncu İki"}

    G = bg.build_graph(cast, name_map)

    assert G.has_edge("nm001", "nm002")
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_build_graph_edge_weight_counts_shared_movies():
    """İki oyuncu 3 farklı filmde birlikte oynadıysa kenar ağırlığı 3 olmalı."""
    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001", "tt002", "tt002", "tt003", "tt003"],
        "nconst": ["nm001", "nm002", "nm001", "nm002", "nm001", "nm002"],
        "category": ["actor", "actress"] * 3,
    })
    name_map = {"nm001": "Oyuncu Bir", "nm002": "Oyuncu İki"}

    G = bg.build_graph(cast, name_map)

    assert G["nm001"]["nm002"]["weight"] == 3


def test_build_graph_no_self_loops_from_duplicate_credits():
    """
    Bazen aynı oyuncu aynı filmde birden fazla kayıtla geçebilir (örn. hem
    'actor' hem 'self' kategorisiyle). Bu durumda kendine kenar (self-loop)
    OLUŞMAMALI.
    """
    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001"],
        "nconst": ["nm001", "nm001"],
        "category": ["actor", "actor"],
    })
    name_map = {"nm001": "Oyuncu Bir"}

    G = bg.build_graph(cast, name_map)

    assert not G.has_edge("nm001", "nm001")
    assert G.number_of_edges() == 0


def test_build_graph_three_actors_same_movie_forms_triangle():
    """3 oyuncu aynı filmde oynadıysa aralarında bir üçgen (3 kenar) oluşmalı."""
    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001", "tt001"],
        "nconst": ["nm001", "nm002", "nm003"],
        "category": ["actor", "actress", "actor"],
    })
    name_map = {}

    G = bg.build_graph(cast, name_map)

    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3  # C(3,2) = 3


def test_build_graph_assigns_name_attribute_with_fallback():
    """İsim eşlemesi yoksa düğüm en azından kendi ID'sini 'name' olarak taşımalı."""
    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001"],
        "nconst": ["nm001", "nm999"],  # nm999 için isim yok
        "category": ["actor", "actress"],
    })
    name_map = {"nm001": "Bilinen İsim"}

    G = bg.build_graph(cast, name_map)

    assert G.nodes["nm001"]["name"] == "Bilinen İsim"
    assert G.nodes["nm999"]["name"] == "nm999"  # fallback: ID'nin kendisi

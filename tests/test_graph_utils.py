"""
test_graph_utils.py
--------------------
graph_utils.py içindeki fonksiyonları küçük, elle hesaplanabilir graflar ve
NetworkX'in ünlü "karate club" veri seti üzerinde test eder.

Çalıştırmak için proje kök klasöründe:
    pytest tests/ -v
"""

import networkx as nx
import pytest

import graph_utils as gu


# ---------- Yardımcı fabrika fonksiyonları (test verisi üretir) ----------

def make_triangle():
    """3 düğümlü tam bağlı bir üçgen: A-B, B-C, A-C."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=1)
    return G


def make_path_graph():
    """Doğrusal bir zincir: A-B-C-D-E (çap = 4)."""
    G = nx.path_graph(["A", "B", "C", "D", "E"])
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


def make_disconnected_graph():
    """Biri 3, diğeri 2 düğümlü iki ayrı bileşen."""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])   # bileşen 1: 3 düğüm
    G.add_edges_from([("X", "Y")])                # bileşen 2: 2 düğüm
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


def make_bridge_graph():
    """
    İki üçgeni tek bir kenarla (köprü) birleştiren graf:
    A-B-C-A (üçgen 1), D-E-F-D (üçgen 2), C-D köprü.
    C ve D birer kesme noktasıdır (articulation point).
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    G.add_edges_from([("D", "E"), ("E", "F"), ("D", "F")])
    G.add_edge("C", "D")
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


# ---------- basic_stats ----------

def test_basic_stats_triangle():
    G = make_triangle()
    stats = gu.basic_stats(G)
    assert stats["n_nodes"] == 3
    assert stats["n_edges"] == 3
    assert stats["is_connected"] is True
    assert stats["n_components"] == 1


def test_basic_stats_disconnected():
    G = make_disconnected_graph()
    stats = gu.basic_stats(G)
    assert stats["n_components"] == 2
    assert stats["is_connected"] is False


# ---------- get_largest_component ----------

def test_get_largest_component_returns_biggest_piece():
    G = make_disconnected_graph()
    largest = gu.get_largest_component(G)
    assert largest.number_of_nodes() == 3
    assert set(largest.nodes()) == {"A", "B", "C"}


def test_get_largest_component_already_connected():
    G = make_triangle()
    result = gu.get_largest_component(G)
    assert result.number_of_nodes() == G.number_of_nodes()


# ---------- degree_sequence ----------

def test_degree_sequence_triangle():
    G = make_triangle()
    degrees = gu.degree_sequence(G)
    # Üçgende her düğümün derecesi 2'dir
    assert sorted(degrees) == [2, 2, 2]


# ---------- centrality_measures ----------

def test_centrality_measures_triangle_is_symmetric():
    """Üçgende tüm düğümler simetrik olduğu için tüm merkezilik skorları eşit olmalı."""
    G = make_triangle()
    centrality = gu.centrality_measures(G)
    degree_scores = {n: v["degree"] for n, v in centrality.items()}
    assert len(set(round(v, 6) for v in degree_scores.values())) == 1


def test_centrality_measures_path_graph_middle_node_highest_betweenness():
    """A-B-C-D-E zincirinde en yüksek betweenness'e sahip olan ortadaki C'dir."""
    G = make_path_graph()
    centrality = gu.centrality_measures(G)
    top_betweenness_node = max(centrality, key=lambda n: centrality[n]["betweenness"])
    assert top_betweenness_node == "C"


# ---------- clustering_metrics ----------

def test_clustering_metrics_triangle_is_fully_clustered():
    """Bir üçgende kümelenme katsayısı tam olarak 1.0 olmalıdır."""
    G = make_triangle()
    cm = gu.clustering_metrics(G)
    assert cm["average_clustering"] == pytest.approx(1.0)
    assert cm["transitivity"] == pytest.approx(1.0)


def test_clustering_metrics_path_graph_has_no_triangles():
    """Bir zincirde hiç üçgen olmadığı için kümelenme katsayısı 0 olmalıdır."""
    G = make_path_graph()
    cm = gu.clustering_metrics(G)
    assert cm["transitivity"] == pytest.approx(0.0)


# ---------- path_metrics ----------

def test_path_metrics_path_graph_diameter():
    """5 düğümlü zincirde çap tam olarak 4 adım olmalıdır (A'dan E'ye)."""
    G = make_path_graph()
    pm = gu.path_metrics(G)
    assert pm["diameter"] == 4


def test_path_metrics_raises_on_disconnected_graph():
    """Bağlantısız bir graf üzerinde çap hesaplamaya çalışmak hata vermelidir."""
    G = make_disconnected_graph()
    with pytest.raises(ValueError):
        gu.path_metrics(G)


# ---------- articulation_points ----------

def test_articulation_points_bridge_graph():
    """İki üçgeni birleştiren köprü grafında C ve D kesme noktası olmalıdır."""
    G = make_bridge_graph()
    points = set(gu.find_articulation_points(G))
    assert points == {"C", "D"}


def test_articulation_points_triangle_has_none():
    """Bir üçgende hiçbir düğüm çıkarılınca graf parçalanmaz."""
    G = make_triangle()
    points = gu.find_articulation_points(G)
    assert points == []


# ---------- shortest_path_between ----------

def test_shortest_path_between_path_graph():
    G = make_path_graph()
    path = gu.shortest_path_between(G, "A", "E")
    assert path == ["A", "B", "C", "D", "E"]


# ---------- robustness_simulation ----------

def test_robustness_simulation_largest_component_shrinks():
    """Düğümler çıkarıldıkça en büyük bileşen boyutu artmamalı (monoton azalan/sabit)."""
    G = make_bridge_graph()
    centrality = gu.centrality_measures(G)
    results = gu.robustness_simulation(G, centrality, metric="betweenness", max_removals=3)

    sizes = [size for _, size in results]
    # Boyutlar hiçbir zaman bir öncekinden büyük olamaz
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))


def test_robustness_simulation_removing_bridge_node_splits_graph():
    """C veya D (kesme noktaları) çıkarılınca en büyük bileşen küçülmelidir."""
    G = make_bridge_graph()
    centrality = gu.centrality_measures(G)
    results = gu.robustness_simulation(G, centrality, metric="betweenness", max_removals=1)
    # Başlangıçta 6 düğüm bağlıydı; en merkezi (C veya D) çıkarılınca en büyük
    # bileşen 3'ten büyük olamaz (iki üçgenden biri).
    assert results[1][1] <= 3


# ---------- common_neighbor_link_predictions ----------

def test_link_prediction_finds_pair_with_common_neighbor():
    """
    A-C ve B-C bağlı ama A-B bağlı değil. A ve B'nin ortak komşusu C olduğu
    için bu çift öneri listesinde çıkmalı.
    """
    G = nx.Graph()
    G.add_edge("A", "C", weight=1)
    G.add_edge("B", "C", weight=1)
    predictions = gu.common_neighbor_link_predictions(G, top_n=5)
    pairs = [frozenset([u, v]) for u, v, _ in predictions]
    assert frozenset(["A", "B"]) in pairs


# ---------- karate club - gerçekçi, bilinen bir graf üzerinde entegrasyon testi ----------

def test_karate_club_integration():
    """
    NetworkX'in ünlü 'karate club' veri seti (34 düğüm, tek bileşen) ile
    tüm fonksiyonların hatasız ve mantıklı sonuçlar ürettiğini doğrular.
    """
    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1

    stats = gu.basic_stats(G)
    assert stats["n_nodes"] == 34
    assert stats["is_connected"] is True

    centrality = gu.centrality_measures(G)
    assert len(centrality) == 34
    # Bilinen bir sonuç: düğüm 33 (veya 0), bu grafta en yüksek dereceli düğümlerdendir
    top_degree_node = max(centrality, key=lambda n: centrality[n]["degree"])
    assert top_degree_node in (0, 33)

    cm = gu.clustering_metrics(G)
    assert 0 <= cm["average_clustering"] <= 1

    pm = gu.path_metrics(G)
    assert pm["diameter"] >= 1

    partition, modularity = gu.louvain_communities(G)
    assert len(partition) == 34
    assert modularity > 0  # bilinen topluluk yapısı nedeniyle pozitif olmalı


# ---------- small_world_metrics ----------

def test_small_world_metrics_returns_expected_keys():
    G = nx.karate_club_graph()
    result = gu.small_world_metrics(G, n_random=5, seed=1)
    assert set(result.keys()) == {
        "C_actual", "L_actual", "C_random_avg", "L_random_avg",
        "sigma", "n_random_trials"
    }
    assert result["n_random_trials"] > 0
    assert result["sigma"] is None or result["sigma"] > 0


def test_small_world_metrics_raises_on_disconnected_graph():
    G = make_disconnected_graph()
    with pytest.raises(ValueError):
        gu.small_world_metrics(G)


def test_small_world_metrics_karate_club_is_small_world():
    """
    Karate club literatürde bilinen bir küçük-dünya ağı örneğidir; sigma
    değeri 1'in belirgin şekilde üzerinde çıkmalıdır.
    """
    G = nx.karate_club_graph()
    result = gu.small_world_metrics(G, n_random=15, seed=7)
    assert result["sigma"] > 1


# ---------- clique_analysis ----------

def test_clique_analysis_triangle_is_one_clique_of_size_three():
    G = make_triangle()
    result = gu.clique_analysis(G)
    assert result["n_cliques"] == 1
    assert result["largest_clique_size"] == 3
    assert sorted(result["top_cliques"][0]) == ["A", "B", "C"]


def test_clique_analysis_bridge_graph_has_two_triangular_cliques():
    """
    İki üçgen ({A,B,C} ve {D,E,F}) artı bunları birleştiren C-D köprü kenarı
    kendi başına maksimal bir klik oluşturur (çünkü C-D hiçbir üçgenin
    parçası değildir) - toplam 3 maksimal klik.
    """
    G = make_bridge_graph()
    result = gu.clique_analysis(G)
    assert result["n_cliques"] == 3
    assert result["largest_clique_size"] == 3
    assert sorted(result["size_distribution"]) == [2, 3, 3]


def test_clique_analysis_path_graph_max_clique_is_an_edge():
    """Bir zincirde (döngü/üçgen yok) en büyük klik sadece 2 düğümlü bir kenardır."""
    G = make_path_graph()
    result = gu.clique_analysis(G)
    assert result["largest_clique_size"] == 2


# ---------- k_core_analysis ----------

def test_k_core_analysis_triangle_is_2_core():
    """Bir üçgende her düğümün derecesi 2'dir, dolayısıyla max_k = 2."""
    G = make_triangle()
    result = gu.k_core_analysis(G)
    assert result["max_k"] == 2
    assert result["main_core_size"] == 3


def test_k_core_analysis_path_graph_is_1_core():
    """Bir zincirin uç düğümlerinin derecesi 1'dir, dolayısıyla max_k = 1."""
    G = make_path_graph()
    result = gu.k_core_analysis(G)
    assert result["max_k"] == 1
    # 1-core = bağlantılı bütün graf (tüm düğümler en az 1 dereceli)
    assert result["main_core_size"] == 5


# ---------- build_bipartite_graph / project_bipartite ----------

def test_bipartite_projection_matches_direct_co_occurrence_graph():
    """
    build_bipartite_graph + project_bipartite ile üretilen oyuncu-oyuncu
    grafı, build_graph.py'daki doğrudan co-occurrence mantığıyla üretilen
    grafla AYNI kenarları vermelidir - iki farklı yöntemin tutarlılığını
    doğrular.
    """
    import pandas as pd

    cast = pd.DataFrame({
        "tconst": ["tt001", "tt001", "tt002", "tt002", "tt002"],
        "nconst": ["nmA", "nmB", "nmB", "nmC", "nmD"],
    })

    B, actor_nodes, movie_nodes = gu.build_bipartite_graph(cast)

    assert set(actor_nodes) == {"nmA", "nmB", "nmC", "nmD"}
    assert set(movie_nodes) == {"tt001", "tt002"}
    assert nx.is_bipartite(B)

    projected = gu.project_bipartite(B, actor_nodes)

    # tt001: nmA-nmB ortak film; tt002: nmB-nmC, nmB-nmD, nmC-nmD ortak film
    expected_edges = {
        frozenset(["nmA", "nmB"]),
        frozenset(["nmB", "nmC"]),
        frozenset(["nmB", "nmD"]),
        frozenset(["nmC", "nmD"]),
    }
    actual_edges = {frozenset(e) for e in projected.edges()}
    assert actual_edges == expected_edges


def test_bipartite_graph_assigns_actor_names():
    import pandas as pd

    cast = pd.DataFrame({"tconst": ["tt001"], "nconst": ["nmA"]})
    name_map = {"nmA": "Test Oyuncu"}

    B, actor_nodes, _ = gu.build_bipartite_graph(cast, name_map)

    assert B.nodes["nmA"]["name"] == "Test Oyuncu"
    assert B.nodes["nmA"]["bipartite"] == 0
    assert B.nodes["tt001"]["bipartite"] == 1

"""
graph_utils.py
--------------
Grafla ilgili tüm saf (pure) hesaplama fonksiyonları burada toplanır.
Bu dosyadaki fonksiyonlar hiçbir I/O (dosya okuma/yazma, print) yapmaz;
sadece bir graf alır ve sonuç döndürür. Bu sayede pytest ile kolayca
test edilebilirler. Yazdırma/kaydetme işleri analyze.py ve
advanced_analysis.py içinde yapılır.
"""

import networkx as nx


def get_largest_component(G):
    """Grafın en büyük bağlantılı bileşenini (connected component) döndürür."""
    if nx.is_connected(G):
        return G.copy()
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def basic_stats(G):
    """Temel graf istatistiklerini bir sözlük olarak döndürür."""
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "n_components": nx.number_connected_components(G),
        "is_connected": nx.is_connected(G),
    }


def degree_sequence(G):
    """Her düğümün derecesini liste olarak döndürür (histogram için)."""
    return [d for _, d in G.degree()]


def centrality_measures(G):
    """
    Dört farklı merkezilik ölçütünü hesaplar ve
    {node: {"degree": ..., "betweenness": ..., "eigenvector": ..., "closeness": ..., "pagerank": ...}}
    formatında birleşik bir sözlük döndürür.
    """
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight="weight")
    closeness_c = nx.closeness_centrality(G, distance="weight")
    pagerank_c = nx.pagerank(G, weight="weight")
    try:
        eigen_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigen_c = nx.eigenvector_centrality_numpy(G, weight="weight")

    combined = {}
    for node in G.nodes():
        combined[node] = {
            "degree": degree_c[node],
            "betweenness": betweenness_c[node],
            "eigenvector": eigen_c[node],
            "closeness": closeness_c[node],
            "pagerank": pagerank_c[node],
        }
    return combined


def clustering_metrics(G):
    """
    Kümelenme katsayılarını döndürür:
    - average_clustering: düğümlerin ortalama yerel kümelenme katsayısı
      ("arkadaşlarımın da birbirini tanıma olasılığı")
    - transitivity: global kümelenme katsayısı (üçgen sayısına dayalı)
    """
    return {
        "average_clustering": nx.average_clustering(G, weight="weight"),
        "transitivity": nx.transitivity(G),
    }


def path_metrics(G):
    """
    Bağlantılı bir graf üzerinde çap (diameter) ve ortalama en kısa yol
    uzunluğunu döndürür. Büyük graflarda çap hesaplaması yavaş olabileceği
    için burada ağırlıksız (unweighted) hesaplanır.
    """
    if not nx.is_connected(G):
        raise ValueError("path_metrics sadece bağlantılı graf üzerinde çalışır. "
                          "Önce get_largest_component ile en büyük bileşeni alın.")
    return {
        "diameter": nx.diameter(G),
        "avg_shortest_path_length": nx.average_shortest_path_length(G),
    }


def assortativity(G):
    """
    Derece asortativite katsayısı: yüksek dereceli düğümler birbirine mi
    bağlanıyor (pozitif değer) yoksa yüksek dereceliler düşük derecelilere
    mi bağlanıyor (negatif değer)?
    """
    return nx.degree_assortativity_coefficient(G, weight="weight")


def find_articulation_points(G):
    """
    Kesme noktalarını (articulation points / cut vertices) bulur.
    Bu düğümler çıkarılırsa graf parçalanır - yani ağın "kritik" oyuncularıdır.
    """
    return list(nx.articulation_points(G))


def shortest_path_between(G, source, target):
    """İki düğüm arasındaki en kısa yolu (node listesi) döndürür."""
    return nx.shortest_path(G, source, target)


def louvain_communities(G):
    """
    Louvain algoritmasıyla topluluk tespiti yapar.
    Döndürür: (partition_dict, modularity_score)
    """
    import community as community_louvain
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    modularity = community_louvain.modularity(partition, G, weight="weight")
    return partition, modularity


def robustness_simulation(G, centrality_dict, metric="betweenness", max_removals=None):
    """
    Düğümleri en yüksek merkezilikten
    başlayarak tek tek çıkarır ve her adımda en büyük bağlantılı bileşenin
    boyutunu kaydeder. "Hedefli saldırı" senaryosunu simüle eder - sosyal
    ağların önemli düğümlere karşı kırılganlığını göstermek için klasik
    bir yöntemdir.
    """
    H = G.copy()
    if max_removals is None:
        max_removals = H.number_of_nodes() - 1

    # Düğümleri merkezilik skoruna göre büyükten küçüğe sırala
    ranked_nodes = sorted(centrality_dict.keys(),
                           key=lambda n: centrality_dict[n][metric], reverse=True)

    results = [(0, len(max(nx.connected_components(H), key=len)))]
    for i, node in enumerate(ranked_nodes[:max_removals], start=1):
        if node in H:
            H.remove_node(node)
        if H.number_of_nodes() == 0:
            results.append((i, 0))
            break
        largest = len(max(nx.connected_components(H), key=len))
        results.append((i, largest))

    return results


def random_failure_simulation(G, seed=42, max_removals=None):
    """
    Karşılaştırma için: düğümleri RASTGELE sırayla çıkarır (hedefli saldırı
    değil, tesadüfi arıza senaryosu). Robustness testinin "kontrol grubu"dur.
    """
    import random
    H = G.copy()
    nodes = list(H.nodes())
    random.seed(seed)
    random.shuffle(nodes)

    if max_removals is None:
        max_removals = len(nodes) - 1

    results = [(0, len(max(nx.connected_components(H), key=len)))]
    for i, node in enumerate(nodes[:max_removals], start=1):
        if node in H:
            H.remove_node(node)
        if H.number_of_nodes() == 0:
            results.append((i, 0))
            break
        largest = len(max(nx.connected_components(H), key=len))
        results.append((i, largest))

    return results


def ego_network(G, node, radius=1):
    """Bir düğümün 'ego ağını' (kendisi + doğrudan komşuları) döndürür."""
    return nx.ego_graph(G, node, radius=radius)


def common_neighbor_link_predictions(G, top_n=10):
    """
    Basit bir link prediction yöntemi: birbirine henüz bağlı OLMAYAN ama
    çok sayıda ortak komşusu (ortak oyuncu arkadaşı) olan düğüm çiftlerini
    bulur. "Bu iki oyuncu henüz birlikte oynamadı ama ortak çevreleri çok
    geniş, gelecekte birlikte çalışabilirler" yorumunu destekler.

    Döndürür: [(node_a, node_b, ortak_komşu_sayısı), ...] büyükten küçüğe sıralı
    """
    scores = []
    non_edges = nx.non_edges(G)
    for u, v in non_edges:
        common = len(list(nx.common_neighbors(G, u, v)))
        if common > 0:
            scores.append((u, v, common))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_n]


def small_world_metrics(G, n_random=10, seed=42):
    import random as _random

    if not nx.is_connected(G):
        raise ValueError("small_world_metrics bağlantılı bir graf gerektirir. "
                          "Önce get_largest_component ile en büyük bileşeni alıyoruz.")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    C_actual = nx.transitivity(G)
    L_actual = nx.average_shortest_path_length(G)

    rng = _random.Random(seed)
    C_randoms, L_randoms = [], []
    attempts = 0
    max_attempts = n_random * 5

    while len(C_randoms) < n_random and attempts < max_attempts:
        attempts += 1
        R = nx.gnm_random_graph(n, m, seed=rng.randint(0, 10**7))
        if not nx.is_connected(R):
            continue  # bağlantısız çıkan rastgele graf örneklerini atla
        C_randoms.append(nx.transitivity(R))
        L_randoms.append(nx.average_shortest_path_length(R))

    if not C_randoms:
        raise RuntimeError(
            "Bağlantılı bir rastgele graf üretilemedi (n_random/seed değiştirmeyi deneyin)."
        )

    C_random_avg = sum(C_randoms) / len(C_randoms)
    L_random_avg = sum(L_randoms) / len(L_randoms)

    sigma = None
    if C_random_avg > 0 and L_random_avg > 0 and L_actual > 0:
        sigma = (C_actual / C_random_avg) / (L_actual / L_random_avg)

    return {
        "C_actual": C_actual,
        "L_actual": L_actual,
        "C_random_avg": C_random_avg,
        "L_random_avg": L_random_avg,
        "sigma": sigma,
        "n_random_trials": len(C_randoms),
    }


def clique_analysis(G, top_n=5):
    """
    Maksimal klikleri (herkesin herkesle bağlı olduğu en büyük alt gruplar)
    bulur. Film ağlarında bir klik genelde "aynı filmin kadrosu" anlamına
    gelir (veya birden fazla filmde üst üste oynayan bir grup).

    Döndürür: {
        "n_cliques": ..., "largest_clique_size": ...,
        "size_distribution": [...], "top_cliques": [[node, node, ...], ...]
    }
    """
    cliques = list(nx.find_cliques(G))
    sizes = [len(c) for c in cliques]
    cliques_sorted = sorted(cliques, key=len, reverse=True)

    return {
        "n_cliques": len(cliques),
        "largest_clique_size": max(sizes) if sizes else 0,
        "size_distribution": sizes,
        "top_cliques": cliques_sorted[:top_n],
    }


def k_core_analysis(G):
    """
    K-core ayrıştırması: bir düğümün "k-core numarası", o düğümün en az k
    dereceye sahip olduğu en büyük alt grafın parçası olduğu k değeridir.
    Ağın en "sıkı bağlı" çekirdek kısmını bulmak için kullanılır.

    Döndürür: {
        "core_numbers": {node: k, ...}, "max_k": ...,
        "main_core_size": ..., "main_core_nodes": [...]
    }
    """
    G_simple = G.copy()
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    core_numbers = nx.core_number(G_simple)
    max_k = max(core_numbers.values()) if core_numbers else 0
    main_core = nx.k_core(G_simple, k=max_k, core_number=core_numbers)

    return {
        "core_numbers": core_numbers,
        "max_k": max_k,
        "main_core_size": main_core.number_of_nodes(),
        "main_core_nodes": list(main_core.nodes()),
    }


def build_bipartite_graph(cast_df, name_map=None):
    """
    Oyuncu-Film İKİ PARÇALI (bipartite) grafını doğrudan ham cast verisinden
    kurar. İki farklı düğüm tipi vardır:
      - Oyuncu düğümleri (bipartite=0)
      - Film düğümleri   (bipartite=1)
    Bir oyuncu bir filmde oynadıysa aralarında kenar vardır.

    Bu, projedeki asıl oyuncu-oyuncu ağının NASIL türetildiğini açıkça
    gösteren ara temsildir (bipartite projection kavramı).

    Döndürür: (B, actor_nodes, movie_nodes)
    """
    B = nx.Graph()
    actor_nodes = cast_df["nconst"].unique().tolist()
    movie_nodes = cast_df["tconst"].unique().tolist()

    B.add_nodes_from(actor_nodes, bipartite=0)
    B.add_nodes_from(movie_nodes, bipartite=1)

    for _, row in cast_df.iterrows():
        B.add_edge(row["nconst"], row["tconst"])

    if name_map:
        for node in actor_nodes:
            B.nodes[node]["name"] = name_map.get(node, node)

    return B, actor_nodes, movie_nodes


def project_bipartite(B, nodes):
    """
    İki parçalı grafı tek tip düğüme göre "projekte eder" (bipartite
    projection). Örneğin actor_nodes verilirse, iki oyuncu ortak en az bir
    filmde oynadıysa aralarında (ortak film sayısı kadar ağırlıklı) bir
    kenar oluşur - build_graph.py'daki actor-actor ağıyla aynı mantık,
    ama NetworkX'in hazır bipartite modülüyle.
    """
    return nx.bipartite.weighted_projected_graph(B, nodes)

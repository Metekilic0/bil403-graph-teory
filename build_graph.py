"""
build_graph.py
----------------
IMDb veri setlerinden (title.basics, title.principals, name.basics) belirli
bir tür/dönemdeki filmleri filtreler ve bu filmlerde birlikte oynayan
oyuncular arasında ağırlıklı bir graf (Actor Co-occurrence Network) oluşturur.

Düğüm (node): Oyuncu (nconst, isim)
Kenar (edge): İki oyuncu en az bir filmde birlikte oynamışsa
Kenar ağırlığı (weight): Kaç filmde birlikte oynadıkları

Filmler rastgele DEĞİL, IMDb oy sayısına (numVotes) göre EN POPÜLER olanlar
seçilir (title.ratings.tsv.gz kullanılabiliyorsa). Bunun sebebi: popüler
filmler genelde tanınmış oyunculara sahiptir, ve tanınmış oyuncular birden
fazla filmde tekrar tekrar karşımıza çıkar - bu da ağın daha az parçalı
(daha bağlantılı) çıkmasını sağlar. title.ratings.tsv.gz yoksa eskisi gibi
rastgele örnekleme yapılır (hiçbir şey kırılmaz, sadece bağlantı kalitesi
daha düşük olur).

Kullanım:
    python build_graph.py --genre "Thriller" --min-year 2005 --max-year 2024 --max-movies 800
"""

import argparse
import itertools
import os
import networkx as nx
import pandas as pd

DATA_DIR = "data"
OUTPUT_DIR = "output"


def parse_args():
    p = argparse.ArgumentParser(description="IMDb verisinden oyuncu ağı oluştur")
    p.add_argument("--genre", type=str, default="Thriller",
                    help="IMDb tür etiketi (örn: Thriller, Mystery, Horror, Crime)")
    p.add_argument("--min-year", type=int, default=2005)
    p.add_argument("--max-year", type=int, default=2024)
    p.add_argument("--max-movies", type=int, default=800,
                    help="Kullanılacak maksimum film sayısı (büyük graf istemiyorsan düşük tut)")
    p.add_argument("--min-votes", type=int, default=0,
                    help="Sadece en az bu kadar IMDb oyu almış filmleri dahil et "
                         "(title.ratings.tsv.gz gerektirir; 0 = filtre yok)")
    p.add_argument("--seed", type=int, default=42,
                    help="Popülerlik verisi YOKSA kullanılan rastgele örnekleme tohumu")
    return p.parse_args()


def resolve_data_file(base_name, required=True):
    """
    Hem sıkıştırılmış (.gz) hem çıkarılmış (düz .tsv) dosyaları otomatik
    tanır. Windows Gezgini uzantıları gizlediği için kullanıcılar genelde
    ikisini birbirine karıştırıyor - bu fonksiyon o sorunu ortadan kaldırır.
    Öncelik .gz dosyasında (daha az yer kaplar); o yoksa düz .tsv'ye bakar.

    required=False ise dosya bulunamadığında hata fırlatmak yerine None döner
    (opsiyonel dosyalar - örn. title.ratings.tsv - için kullanılır).
    """
    gz_path = os.path.join(DATA_DIR, base_name + ".gz")
    tsv_path = os.path.join(DATA_DIR, base_name)

    if os.path.exists(gz_path):
        return gz_path, "gzip"
    if os.path.exists(tsv_path):
        return tsv_path, None
    if not required:
        return None, None
    raise FileNotFoundError(
        f"'{base_name}' (veya '{base_name}.gz') bulunamadı. "
        f"README.md'deki linklerden indirip data/ klasörüne koy."
    )


def load_ratings():
    """
    title.ratings.tsv.gz dosyasını okur (varsa). Bu dosya opsiyoneldir;
    yoksa None döner ve pipeline eskisi gibi rastgele örneklemeye düşer.
    """
    path, compression = resolve_data_file("title.ratings.tsv", required=False)
    if path is None:
        return None

    print(f"{os.path.basename(path)} okunuyor (popülerlik/oy verisi)...")
    ratings = pd.read_csv(
        path, sep="\t", compression=compression, na_values="\\N",
        usecols=["tconst", "numVotes"], dtype={"tconst": str, "numVotes": "Int64"}
    )
    return dict(zip(ratings["tconst"], ratings["numVotes"]))


def load_filtered_movies(genre, min_year, max_year, max_movies, min_votes=0, seed=42):
    path, compression = resolve_data_file("title.basics.tsv")
    print(f"{os.path.basename(path)} okunuyor (bu biraz zaman alabilir)...")

    # Büyük dosyayı parça parça (chunk) okuyup filtreliyoruz ki bellek şişmesin
    chunks = pd.read_csv(
        path, sep="\t", compression=compression, na_values="\\N",
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
        dtype=str, chunksize=200_000
    )

    filtered_parts = []
    for chunk in chunks:
        chunk = chunk[chunk["titleType"] == "movie"]
        chunk = chunk.dropna(subset=["startYear", "genres"])
        chunk["startYear"] = pd.to_numeric(chunk["startYear"], errors="coerce")
        chunk = chunk[(chunk["startYear"] >= min_year) & (chunk["startYear"] <= max_year)]
        chunk = chunk[chunk["genres"].str.contains(genre, case=False, na=False)]
        if not chunk.empty:
            filtered_parts.append(chunk)

    movies = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()
    print(f"Filtre sonrası bulunan film sayısı: {len(movies)}")

    if movies.empty:
        return movies

    ratings_map = load_ratings()

    if ratings_map is not None:
        movies["numVotes"] = movies["tconst"].map(ratings_map).fillna(0).astype(int)

        if min_votes > 0:
            before = len(movies)
            movies = movies[movies["numVotes"] >= min_votes]
            print(f"--min-votes {min_votes} filtresiyle {before} -> {len(movies)} filme indirildi.")

        if len(movies) > max_movies:
            # Rastgele değil, EN ÇOK OY ALAN (en popüler) filmleri seç.
            # Popüler filmler tanınmış oyunculara sahiptir, bu da ağın
            # daha bağlantılı çıkmasını sağlar.
            movies = movies.sort_values("numVotes", ascending=False).head(max_movies)
            print(f"En popüler {max_movies} film seçildi (numVotes'a göre sıralı).")
    else:
        print("UYARI: title.ratings.tsv(.gz) bulunamadı, popülerlik filtresi uygulanamıyor. "
              "Rastgele örnekleme yapılacak (daha parçalı bir ağ çıkabilir).")
        if len(movies) > max_movies:
            movies = movies.sample(n=max_movies, random_state=seed)
            print(f"--max-movies ile {max_movies} filme rastgele indirildi.")

    return movies


def load_cast_for_movies(movie_ids):
    path, compression = resolve_data_file("title.principals.tsv")
    print(f"{os.path.basename(path)} okunuyor (büyük dosya, sabırlı ol)...")
    movie_id_set = set(movie_ids)

    chunks = pd.read_csv(
        path, sep="\t", compression=compression, na_values="\\N",
        usecols=["tconst", "nconst", "category"],
        dtype=str, chunksize=500_000
    )

    cast_parts = []
    for chunk in chunks:
        chunk = chunk[chunk["tconst"].isin(movie_id_set)]
        chunk = chunk[chunk["category"].isin(["actor", "actress"])]
        if not chunk.empty:
            cast_parts.append(chunk)

    cast = pd.concat(cast_parts, ignore_index=True) if cast_parts else pd.DataFrame()
    print(f"Bulunan oyuncu-film kaydı: {len(cast)}")
    return cast


def load_actor_names(nconsts):
    path, compression = resolve_data_file("name.basics.tsv")
    print(f"{os.path.basename(path)} okunuyor...")
    nconst_set = set(nconsts)

    chunks = pd.read_csv(
        path, sep="\t", compression=compression, na_values="\\N",
        usecols=["nconst", "primaryName"], dtype=str, chunksize=200_000
    )

    name_parts = []
    for chunk in chunks:
        chunk = chunk[chunk["nconst"].isin(nconst_set)]
        if not chunk.empty:
            name_parts.append(chunk)

    names = pd.concat(name_parts, ignore_index=True) if name_parts else pd.DataFrame()
    return dict(zip(names["nconst"], names["primaryName"]))


def build_graph(cast_df, name_map):
    G = nx.Graph()

    # Her film için oynayan oyuncuları grupla, ikili kombinasyonlar arasına kenar çek
    grouped = cast_df.groupby("tconst")["nconst"].apply(list)

    for actors in grouped:
        actors = list(set(actors))  # aynı filmde tekrar eden kayıtları temizle
        for a, b in itertools.combinations(sorted(actors), 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    # Düğümlere isim ekle
    for node in G.nodes():
        G.nodes[node]["name"] = name_map.get(node, node)

    return G


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    movies = load_filtered_movies(args.genre, args.min_year, args.max_year,
                                   args.max_movies, min_votes=args.min_votes, seed=args.seed)
    if movies.empty:
        raise ValueError("Seçilen tür/yıl aralığında film bulunamadı. Parametreleri gevşet.")

    cast = load_cast_for_movies(movies["tconst"].tolist())
    if cast.empty:
        raise ValueError("Bu filmler için oyuncu verisi bulunamadı.")

    name_map = load_actor_names(cast["nconst"].unique().tolist())

    print("Graf oluşturuluyor...")
    G = build_graph(cast, name_map)
    print(f"Graf tamamlandı: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar")

    out_path = os.path.join(OUTPUT_DIR, "actor_graph.graphml")
    nx.write_graphml(G, out_path)
    print(f"Graf kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

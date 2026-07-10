# Film Oyuncu İşbirliği Ağı - Graf Teorisi Analizi

IMDb verileri kullanılarak belirli bir film türündeki (varsayılan: Thriller)
filmlerde birlikte oynayan oyuncular arasında bir işbirliği ağı (graf) kurar
ve bu ağ üzerinde kapsamlı graf teorisi ve sosyal ağ analizi yöntemleri
uygular: merkezilik ölçütleri, topluluk tespiti, küçük-dünya testi, ağ
dayanıklılığı, klik/k-core analizi, bipartite graf projeksiyonu ve çoklu tür
karşılaştırması.

## Özellikler

- **Ağ kurulumu**: Oyuncu-film iki parçalı (bipartite) grafının oyuncu-oyuncu
  ağına projeksiyonu; kenar ağırlığı = birlikte oynanan film sayısı
- **Merkezilik analizi**: Degree, betweenness, eigenvector, closeness, PageRank
- **Topluluk tespiti**: Louvain algoritması ile modularity tabanlı kümeleme
- **Yapısal analiz**: Kümelenme katsayısı, çap, ortalama yol uzunluğu, derece
  asortativitesi, kesme noktaları (articulation points)
- **Küçük-dünya testi**: Rastgele graflarla karşılaştırmalı sigma (σ) istatistiği
- **Ağ dayanıklılığı**: Hedefli saldırı vs. rastgele arıza simülasyonu
- **Klik ve k-core analizi**: Maksimal klikler ve çekirdek yapı tespiti
- **Link prediction**: Basit ortak-komşu tabanlı öneri sistemi
- **Çoklu tür karşılaştırması**: Farklı film türlerinin ağ yapısını karşılaştırma
- **Bipartite doğrulama**: Projeksiyon yönteminin doğrudan yöntemle tutarlılığını test etme
- **45 birim/entegrasyon testi**: Gerçek veriye ihtiyaç duymadan çalışan pytest test paketi
- **Jupyter Notebook**: Tüm analizleri tek dosyada birleştiren, çalıştırılabilir notebook

## Proje Yapısı

```
.
├── data/                        # IMDb ham verisi (bkz. Veri İndirme)
├── output/                      # Üretilen grafikler, CSV ve raporlar
├── graph_utils.py                # Saf graf metrik fonksiyonları
├── build_graph.py                # Veri okuma, filtreleme ve ağ oluşturma
├── analyze.py                    # Temel analizler (centrality, Louvain)
├── advanced_analysis.py          # İleri analizler (robustness, ego, link prediction)
├── structural_analysis.py        # Yapısal analizler (small-world, clique, k-core)
├── compare_genres.py             # Çoklu tür karşılaştırması
├── bipartite_analysis.py         # Bipartite graf + projeksiyon doğrulaması
├── BIL403_donem_projesi.ipynb    # Tüm analizleri birleştiren Jupyter notebook
├── tests/
│   ├── test_graph_utils.py
│   ├── test_build_graph.py
│   └── test_integration.py
└── requirements.txt
```

## Kurulum

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Veri İndirme

Bu proje [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
kaynağını kullanır. Aşağıdaki dosyaları indirip `data/` klasörüne yerleştirin:

- [title.basics.tsv.gz](https://datasets.imdbws.com/title.basics.tsv.gz)
- [title.principals.tsv.gz](https://datasets.imdbws.com/title.principals.tsv.gz)
- [name.basics.tsv.gz](https://datasets.imdbws.com/name.basics.tsv.gz)
- [title.ratings.tsv.gz](https://datasets.imdbws.com/title.ratings.tsv.gz) *(opsiyonel, popülerlik filtresi için önerilir)*

`title.ratings.tsv.gz` mevcutsa `build_graph.py`, filmleri rastgele değil
IMDb oy sayısına göre en popüler olanlardan seçer; bu da daha bağlantılı bir
ağ elde edilmesini sağlar. Bu dosya olmadan da çalışır, rastgele örneklemeye
döner.

## Kullanım

### 1. Ağı Oluştur

```bash
python build_graph.py --genre "Thriller" --min-year 2005 --max-year 2024 --max-movies 800
```

| Parametre | Açıklama |
|---|---|
| `--genre` | IMDb tür etiketi (Thriller, Comedy, Horror, Crime...) |
| `--min-year` / `--max-year` | Yıl aralığı filtresi |
| `--max-movies` | Kullanılacak maksimum film sayısı |
| `--min-votes` | (opsiyonel) Minimum IMDb oy sayısı eşiği |

Çıktı: `output/actor_graph.graphml`

### 2. Temel Analiz

```bash
python analyze.py
```

Derece dağılımı, beş merkezilik ölçütü, en kısa yol örneği ve Louvain
topluluk tespiti üretir (`output/degree_distribution.png`,
`output/communities.png`, `output/centrality_scores.csv`).

### 3. İleri Analizler

```bash
python advanced_analysis.py
```

Kümelenme katsayısı, çap, derece asortativitesi, kesme noktaları, ağ
dayanıklılık testi (`output/robustness_test.png`), ego ağı ve link
prediction üretir.

### 4. Yapısal Analizler

```bash
python structural_analysis.py
```

Küçük-dünya (sigma) testi, klik analizi (`output/clique_distribution.png`)
ve k-core ayrıştırması (`output/k_core.png`) üretir.

### 5. Tür Karşılaştırması

```bash
python compare_genres.py --genres Thriller Comedy Horror --min-year 2005 --max-year 2024 --max-movies 150
```

Birden fazla türün ağ yapısını karşılaştırmalı tablo (`output/genre_comparison.csv`)
ve grafik (`output/genre_comparison.png`) olarak sunar.

### 6. Bipartite Doğrulama

```bash
python bipartite_analysis.py --genre Thriller --min-year 2005 --max-year 2024 --max-movies 60
```

Oyuncu-film bipartite grafını kurar, oyuncu-oyuncu ağına projekte eder ve
`build_graph.py`'ın doğrudan ürettiği ağla tutarlılığını doğrular.

### 7. Testleri Çalıştır

```bash
pytest tests/ -v
```

### 8. Notebook

```bash
jupyter notebook BIL403_donem_projesi.ipynb
```

Notebook, yukarıdaki tüm script'lerle aynı `build_graph.py` ve
`graph_utils.py` modüllerini kullanır; sonuçlar birebir tutarlıdır.

## Gereksinimler

Bkz. `requirements.txt` (pandas, networkx, matplotlib, python-louvain, numpy,
pytest, jupyter).

## Veri Kaynağı ve Lisans

Veri seti [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
üzerinden ücretsiz olarak sağlanmaktadır ve yalnızca kişisel/eğitim amaçlı
kullanım için lisanslıdır.

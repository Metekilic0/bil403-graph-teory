# Film Türü Oyuncu Ağı - Graf Teorisi Projesi

Belirli bir film türündeki (varsayılan: Psikolojik Gerilim / Mystery) filmlerde
birlikte oynayan oyuncular arasında bir ağ (graf) kurup; en kısa yol, merkezilik
(centrality) ölçütleri ve topluluk tespiti (Louvain) analizleri yapar.

> **BİL403/503 dönem projesi teslimi için önemli:** Proje şartnamesi kodların
> ayrıca bir **IPython (.ipynb)** dosyası olarak da yüklenmesini istiyor.
> `BIL403_donem_projesi.ipynb` dosyası, tüm analizleri (temel + ileri +
> yapısal + bipartite) tek bir notebook'ta, bol açıklamalı şekilde birleştirir.
> Veri indirildikten sonra Jupyter'da baştan sona çalıştırıp çıktılarıyla
> birlikte (rapora ekleyeceğin grafiklerle) teslim edebilirsin. Ayrı `.py`
> script'leri de dursun - notebook onları import ediyor, ikisi birbiriyle
> tutarlı.

## 1. Kurulum

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Veri İndirme (IMDb Datasets - ücretsiz, API key gerekmez)

Aşağıdaki linklerden dosyaları indirip `data/` klasörüne koy:

- https://datasets.imdbws.com/title.basics.tsv.gz
- https://datasets.imdbws.com/title.principals.tsv.gz
- https://datasets.imdbws.com/name.basics.tsv.gz
- https://datasets.imdbws.com/title.ratings.tsv.gz  **(yeni - popülerlik filtresi için)**

`title.ratings.tsv.gz` opsiyoneldir ama **şiddetle önerilir**: bu dosya
sayesinde script, binlerce film arasından RASTGELE değil, EN POPÜLER
(en çok IMDb oyu almış) filmleri seçer. Popüler filmler genelde tanınmış
oyunculara sahiptir, bu da ağın daha az parçalı (daha bağlantılı) çıkmasını
sağlar - önceki denemede karşılaştığımız "259 ayrı bileşen" sorununu
büyük ölçüde azaltır. Bu dosya olmadan da çalışır, sadece eski (rastgele)
davranışa döner.

Not: Bu dosyalar büyük (title.principals ~800MB açıldığında). Kod, gereksiz
satırları erkenden filtreleyerek belleği yönetir; yine de indirme biraz
zaman alabilir.

## 3. Grafı Oluştur

```bash
python build_graph.py --genre "Thriller" --min-year 2005 --max-year 2024 --max-movies 800
```

Parametreler:
- `--genre`: IMDb tür etiketi (örn. Thriller, Mystery, Horror, Crime, Comedy...)
- `--min-year` / `--max-year`: filtrelenecek yıl aralığı
- `--max-movies`: kaç film kullanılacağı (fazlası = daha büyük ama yavaş graf).
  `title.ratings.tsv.gz` mevcutsa bunlar EN POPÜLER filmlerdir, yoksa rastgele seçilir.
- `--min-votes`: (opsiyonel) sadece en az bu kadar oy almış filmleri dahil et,
  örn. `--min-votes 1000` çok bilinmeyen/niş yapımları eler

Önceki denemede 300 rastgele filmle graf 259 parçaya bölünmüştü (en büyük
bileşen sadece 52 kişi). `--max-movies 800` ile popülerlik filtresi
birleşince çok daha bağlantılı, tek parçaya yakın bir ağ beklenir - ama
yine de %100 garantisi yok, IMDb verisinin doğası gereği (farklı ülke/dil
endüstrileri) bir miktar parçalanma normaldir. Raporun "Tartışma" kısmında
bunu bir bulgu olarak yorumlayabilirsin.

Bu adım `output/actor_graph.graphml` dosyasını üretir.

## 4. Analiz Et

```bash
python analyze.py
```

Bu adım şunları yapar ve `output/` klasörüne kaydeder:
- Temel graf istatistikleri (düğüm/kenar sayısı, yoğunluk)
- Derece dağılımı grafiği (degree_distribution.png)
- En yüksek merkeziliğe sahip oyuncular (degree, betweenness, eigenvector)
- İki oyuncu arasında en kısa yol örneği
- Louvain ile topluluk tespiti + modularity skoru
- Topluluklara göre renklendirilmiş graf görselleştirmesi (communities.png)

## 5. İleri Analizler (opsiyonel ama önerilir)

```bash
python advanced_analysis.py
```

`analyze.py` çalıştırıldıktan sonra bu script ek olarak şunları üretir:

- **Kümelenme katsayısı** (clustering coefficient / transitivity)
- **Çap ve ortalama en kısa yol** uzunluğu
- **Derece asortativitesi** (popüler oyuncular birbirleriyle mi çalışıyor?)
- **Kesme noktaları** (articulation points) - hangi oyuncu çıkarsa ağ parçalanır
- **Ağ dayanıklılık testi** (`robustness_test.png`) - en merkezi oyuncuları
  sırayla çıkarma (hedefli saldırı) ile rastgele çıkarmayı karşılaştırır
- **Ego ağı** (`ego_network.png`) - en merkezi oyuncunun doğrudan bağlantıları
- **Link prediction** - henüz birlikte oynamamış ama çok ortak bağlantısı olan
  oyuncu çiftleri (basit "ortak komşu" yöntemiyle)

Sonuçlar `output/advanced_analysis_report.txt` dosyasına da kaydedilir.

## 6. Testleri Çalıştırma

Proje, `graph_utils.py` ve `build_graph.py` içindeki fonksiyonlar için
gerçek IMDb verisine ihtiyaç duymayan (sahte/küçük graf ve DataFrame'lerle
çalışan) unit testler içerir:

```bash
pytest tests/ -v
```

26 test, temel graf işlemlerini (kenar oluşturma, ağırlık sayma, en kısa
yol, kesme noktası tespiti, kümelenme katsayısı, dayanıklılık simülasyonu
vb.) NetworkX'in bilinen "karate club" veri seti dahil çeşitli senaryolarda
doğrular. Bu testleri rapora "kodun doğruluğu birim testlerle doğrulanmıştır"
şeklinde referans verebilirsin.

## Proje Yapısı

```
403/
├── data/                        # IMDb ham verisi (indirmen gerekiyor)
├── output/                      # Üretilen grafikler, CSV ve raporlar
├── graph_utils.py                # Saf graf metrik fonksiyonları (test edilebilir)
├── build_graph.py                # Veri okuma + graf oluşturma
├── analyze.py                    # Temel analizler
├── advanced_analysis.py          # İleri analizler (robustness, ego, link prediction...)
├── tests/
│   ├── test_graph_utils.py
│   └── test_build_graph.py
├── requirements.txt
└── README.md
```

## 5. İleri Analizler (opsiyonel ama önerilir)

```bash
python advanced_analysis.py
```

`analyze.py` çalıştırıldıktan sonra bu script ek olarak şunları üretir:

- **Kümelenme katsayısı** (clustering coefficient / transitivity)
- **Çap ve ortalama en kısa yol** uzunluğu
- **Derece asortativitesi** (popüler oyuncular birbirleriyle mi çalışıyor?)
- **Kesme noktaları** (articulation points) - hangi oyuncu çıkarsa ağ parçalanır
- **Ağ dayanıklılık testi** (`robustness_test.png`) - en merkezi oyuncuları
  sırayla çıkarma (hedefli saldırı) ile rastgele çıkarmayı karşılaştırır
- **Ego ağı** (`ego_network.png`) - en merkezi oyuncunun doğrudan bağlantıları
- **Link prediction** - henüz birlikte oynamamış ama çok ortak bağlantısı olan
  oyuncu çiftleri (basit "ortak komşu" yöntemiyle)

Sonuçlar `output/advanced_analysis_report.txt` dosyasına da kaydedilir.

## 6. Yapısal Analizler (küçük-dünya, klik, k-core)

```bash
python structural_analysis.py
```

`analyze.py` çalıştırıldıktan sonra çalıştırılır. Şunları üretir:

- **Küçük-dünya (small-world) testi** - ağı aynı boyuttaki rastgele graflarla
  karşılaştırarak sigma (σ) skorunu hesaplar. σ > 1 ise ağ gerçekten
  "küçük dünya" özelliği taşıyor demektir (yüksek kümelenme + kısa yol
  uzunluğu bir arada).
- **Klik (clique) analizi** (`clique_distribution.png`) - "herkesin herkesle
  bağlı olduğu" en büyük oyuncu grupları (genelde tek bir filmin kadrosuna
  karşılık gelir).
- **K-core ayrıştırması** (`k_core.png`) - ağın en "sıkı bağlı" çekirdek
  kısmını bulur ve görselleştirir.

Sonuçlar `output/structural_analysis_report.txt` dosyasına kaydedilir.

## 7. Farklı Türlerin Karşılaştırılması

```bash
python compare_genres.py --genres Thriller Comedy Horror --min-year 2005 --max-year 2024 --max-movies 150
```

Birden fazla tür için ayrı ayrı ağ kurup (her biri için IMDb verisini yeniden
okur, biraz zaman alır) yoğunluk, kümelenme, en büyük bileşen boyutu ve
ortalama yol uzunluğu gibi metrikleri karşılaştırmalı bir tabloda
(`output/genre_comparison.csv`) ve grafiklerde (`output/genre_comparison.png`)
sunar. "Komedi ağları mı daha kümelenmiş, gerilim ağları mı?" gibi
sorulara doğrudan cevap verir.

## 8. Bipartite (Oyuncu-Film) Graf ve Projeksiyon

```bash
python bipartite_analysis.py --genre Thriller --min-year 2005 --max-year 2024 --max-movies 60
```

Projenin başından beri kullanılan oyuncu-oyuncu ağının aslında NASIL
türetildiğini açıkça gösterir:

1. Önce iki farklı düğüm tipi olan bir **bipartite graf** kurar (Oyuncular +
   Filmler)
2. Bunu oyuncu-oyuncu ağına **projekte eder** ve `build_graph.py`'ın
   doğrudan ürettiği grafla birebir aynı olduğunu doğrular (tutarlılık
   kontrolü ekranda "EVET" olarak görünür)
3. Bonus olarak **film-film ağına** da projekte eder (hangi iki film en çok
   ortak oyuncu paylaşıyor?)
4. Küçük bir örneği iki renkli görsel olarak kaydeder (`bipartite_sample.png`)
   - raporunda "bipartite graf nedir" diye açıklarken kullanabileceğin somut
   bir görsel

## 9. Testleri Çalıştırma

Proje, gerçek IMDb verisine ihtiyaç duymayan (sahte/küçük graf ve
DataFrame'lerle çalışan) **41 unit ve entegrasyon testi** içerir:

```bash
pytest tests/ -v
```

Test dosyaları:
- `tests/test_graph_utils.py` - tüm graf metrik fonksiyonları (centrality,
  clustering, robustness, small-world, clique, k-core, bipartite projeksiyon
  vb.), NetworkX'in bilinen "karate club" veri seti dahil
- `tests/test_build_graph.py` - veri okuma/filtreleme ve graf oluşturma
  mantığı, sahte DataFrame'lerle
- `tests/test_integration.py` - sahte ama gerçek IMDb formatında `.tsv.gz`
  dosyaları oluşturup TÜM pipeline'ı (build_graph -> compare_genres ->
  bipartite) uçtan uca test eder

Bu testleri rapora "kodun doğruluğu birim ve entegrasyon testleriyle
doğrulanmıştır" şeklinde referans verebilirsin.

## Proje Yapısı

```
403/
├── data/                        # IMDb ham verisi (indirmen gerekiyor)
├── output/                      # Üretilen grafikler, CSV ve raporlar
├── graph_utils.py                # Saf graf metrik fonksiyonları (test edilebilir)
├── build_graph.py                # Veri okuma + graf oluşturma
├── analyze.py                    # Temel analizler (centrality, Louvain)
├── advanced_analysis.py          # İleri analizler (robustness, ego, link prediction...)
├── structural_analysis.py        # Yapısal analizler (small-world, clique, k-core)
├── compare_genres.py             # Çoklu tür karşılaştırması
├── bipartite_analysis.py         # Bipartite graf + projeksiyon
├── tests/
│   ├── test_graph_utils.py
│   ├── test_build_graph.py
│   └── test_integration.py
├── requirements.txt
└── README.md
```

## Önerilen Çalıştırma Sırası

```bash
pip install -r requirements.txt
python build_graph.py --genre "Thriller" --min-year 2005 --max-year 2024 --max-movies 800
python analyze.py
python advanced_analysis.py
python structural_analysis.py
python compare_genres.py
python bipartite_analysis.py
pytest tests/ -v
```

**Rapor/teslim için:** Yukarıdakileri script olarak çalıştırdıktan sonra (veya
onun yerine) `jupyter notebook` komutuyla `BIL403_donem_projesi.ipynb`'ı aç ve
"Run All" ile baştan sona çalıştır. Notebook aynı `build_graph.py` ve
`graph_utils.py` modüllerini kullanır, yani script'lerle notebook'un sonuçları
birebir tutarlıdır. Notebook'taki grafik çıktılarını doğrudan rapora
(Bulgular bölümü) kopyalayabilirsin.

## Rapor için Öneriler

- Neden bu tür/dönem seçildiğini kısaca açıkla (veri boyutu makul kalsın diye
  filtrelemenin gerekçesi).
- Derece dağılımının şeklini yorumla (çoğu sosyal ağda "az sayıda çok bağlantılı
  düğüm, çok sayıda az bağlantılı düğüm" görülür - scale-free benzeri yapı).
- Merkezilik ölçütlerinin farkını örnekle anlat: Degree centrality en çok
  ortak oyuncuya sahip olanı, betweenness "köprü" konumundaki oyuncuyu,
  eigenvector ise "önemli kişilerle bağlantılı" oyuncuyu gösterir.
- Louvain sonuçlarında bulunan toplulukların gerçek hayatta neye karşılık
  geldiğini (aynı yönetmenle çalışan kadro, aynı dönem/stüdyo oyuncuları vb.)
  yorumlaman projeye değer katar.
- **Kesme noktaları** ile **ağ dayanıklılık testi**ni birlikte anlat: kesme
  noktası olarak çıkan oyuncuların, robustness grafiğinde de "hedefli
  saldırı" eğrisini hızlıca düşüren isimler olup olmadığını karşılaştır -
  bu iki analiz birbirini doğrular ve rapora güçlü bir bulgu katar.
- **Asortativite** sonucunu, topluluk yapısıyla birlikte yorumla: negatif
  çıkması genelde "yıldız oyuncu + onun etrafındaki daha az bilinen kadro"
  yapısına işaret eder, ki film endüstrisinde beklenen bir örüntüdür.
- **Link prediction** kısmı öğretmenin ilgisini çekebilecek bir bölüm:
  "bu ağın gelecekte nasıl büyüyebileceği" üzerine kısa bir yorum ekleyerek
  projeyi sadece betimsel değil, öngörüsel bir boyuta da taşıyabilirsin.
- **Küçük-dünya testi**ni raporun "özet/sonuç" kısmında kullan: sigma
  skorunun literatürdeki "gerçek sosyal ağlar küçük-dünya özelliği taşır"
  iddiasını somut bir sayıyla doğrulayıp doğrulamadığını tartış.
- **Klik analizi** ile **Louvain toplulukları**nı karşılaştır: eğer bulunan
  topluluklar neredeyse birebir kliklerle örtüşüyorsa, bu ağın "birbirinden
  kopuk, sıkı bağlı küçük film kadrolarından" oluştuğunu güçlü şekilde
  destekler.
- **K-core** sonucu ile **kesme noktaları**nı birlikte yorumla: ana
  çekirdekteki oyuncular genelde aynı zamanda kesme noktası da olabilir -
  bu, "ağın hem en yoğun hem de en kritik bölgesi aynı kişiler" gibi güçlü
  bir bulgu sunar.
- **Tür karşılaştırması** raporun en "özgün" bölümü olabilir: sadece tek
  bir türü değil, birden fazla türü incelediğini göstererek analitik
  derinliğini artırır. "Neden bu fark var?" sorusuna kısa bir yorum ekle
  (örn. komedi filmleri genelde daha küçük ansambl kadrolarla çekilir,
  gerilim filmleri ise daha büyük yapımlarda geniş kadrolara sahip olabilir).
- **Bipartite graf** bölümünü raporun metodoloji kısmında kullan: "aslında
  oyuncu-oyuncu ağı, oyuncu-film iki parçalı grafının projeksiyonudur"
  şeklinde açıklarsan, dersin "graf teorisi" ayağını daha güçlü
  gösterirsin - bu tam da hocanın görmek isteyeceği türden bir kavramsal
  derinlik.

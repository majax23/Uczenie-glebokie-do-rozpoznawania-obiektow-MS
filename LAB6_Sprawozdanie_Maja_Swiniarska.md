# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 6. Warstwa łącząca (pooling) i pełna CNN

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 18.11.2026

## 1. Cel ćwiczenia

Zbadanie, jak sposób redukcji wymiaru przestrzennego wpływa na architekturę (liczba parametrów, rozmiar mapy cech) i accuracy klasyfikacji CNN na FashionMNIST.

- **Zakres obowiązkowy:** porównanie max poolingu 2×2, average poolingu 2×2 i splotu 3×3 z krokiem 2 w sieci z dwoma blokami i zakończeniem GAP + Linear.
- **Zakres dodatkowy:**
  - okno poolingu 2×2 vs 3×3,
  - głowica GAP + Linear vs Flatten + Linear,
  - liczba bloków: 1, 2 i 3.

## 2. Metoda i implementacja

[Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB6_Zadanie_Maja_Swiniarska.ipynb)

**Narzędzia:** PyTorch 2.14.1, torchvision 0.29.1, Google Colab (Tesla T4).

**Zbiór:** FashionMNIST, oficjalny podział 60 000 / 10 000, normalizacja (0.2860, 0.3530), bez augmentacji.

**Architektura:** to klasa `CNN` z ćwiczenia, sparametryzowana:

```python
def block(cin, cout):
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
    if pool == "max":
        layers.append(nn.MaxPool2d(pool_k, stride=2, padding=(pool_k - 1) // 2))
    elif pool == "avg":
        layers.append(nn.AvgPool2d(pool_k, stride=2, padding=(pool_k - 1) // 2))
    elif pool == "stride":
        layers += [nn.Conv2d(cout, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
    return nn.Sequential(*layers)
```

- Kanały w kolejnych blokach: 32 -> 64 -> 128.
- Głowica: `AdaptiveAvgPool2d(1) -> Flatten -> Linear(C, 10)` (GAP) albo `Flatten -> Linear(C·H·W, 10)`.
- Rozmiar mapy przed głowicą wyznaczam przebiegiem próbnym.

**Konfiguracja treningu** (identyczna dla wszystkich 14 wariantów):

| Parametr | Wartość |
|---|---|
| Optymalizator | Adam, lr = 1e-3 |
| Funkcja straty | entropia krzyżowa |
| Epoki | 10 |
| Batch | 128 |
| Seed | 0 (te same wagi startowe dla tej samej architektury i ta sama kolejność batchy) |

Accuracy treningową i testową liczę po każdej epoce na całych zbiorach, w trybie `eval`. W tabelach podaję wartości po ostatniej epoce. Kolumna „maks.\*” to maksimum krzywej testowej, czyli wybór modelu na zbiorze testowym; podaję ją tylko poglądowo.

## 3. Wyniki

### 3.1 Zakres obowiązkowy

| Wariant | Parametry | Mapa przed GAP [C, H, W] | Train acc [%] | Test acc [%] | Przerwa [p.p.] | Maks. test acc\* [%] (epoka) |
|---|---:|---|---:|---:|---:|---|
| max pooling 2×2 | 19 658 | [64, 7, 7] | 84.56 | 83.60 | 0.96 | 84.74 (9) |
| average pooling 2×2 | 19 658 | [64, 7, 7] | 80.59 | 80.27 | 0.33 | 82.24 (8) |
| splot 3×3, krok 2 | 66 026 | [64, 7, 7] | 93.21 | **89.82** | 3.39 | 89.82 (10) |

Różnice względem max poolingu:
- **avg:** −3.33 p.p. przy tej samej liczbie parametrów.
- **stride:** +6.22 p.p. przy +46 368 parametrach (×3.36). Dodatkowe parametry to dwa sploty 3×3 (9 248 + 36 928) i dwie warstwy BN (64 + 128).

<img width="884" height="667" alt="krzywe_obowiazkowe" src="https://github.com/user-attachments/assets/94b0a039-7e3a-40b7-9570-ad78da91b71b" />


### 3.2 Okno poolingu 2×2 vs 3×3 (krok 2, dla 3×3 padding 1)

| Wariant | Parametry | Mapa przed GAP | Train acc [%] | Test acc [%] | Przerwa [p.p.] | Maks.\* [%] (epoka) |
|---|---:|---|---:|---:|---:|---|
| max 2×2 | 19 658 | [64, 7, 7] | 84.56 | 83.60 | 0.96 | 84.74 (9) |
| max 3×3 | 19 658 | [64, 7, 7] | 84.04 | 82.71 | 1.33 | 85.93 (6) |
| avg 2×2 | 19 658 | [64, 7, 7] | 80.59 | 80.27 | 0.33 | 82.24 (8) |
| avg 3×3 | 19 658 | [64, 7, 7] | 81.92 | 81.69 | 0.23 | 81.69 (10) |

<img width="1784" height="667" alt="krzywe_A_okno" src="https://github.com/user-attachments/assets/f4a91b43-9a0e-4140-ae78-48b13fc904f1" />


### 3.3 Głowica: GAP + Linear vs Flatten + Linear

| Wariant | Parametry | Parametry głowicy | Train acc [%] | Test acc [%] | Przerwa [p.p.] | Maks.\* [%] (epoka) |
|---|---:|---:|---:|---:|---:|---|
| max + GAP | 19 658 | 650 | 84.56 | 83.60 | 0.96 | 84.74 (9) |
| max + Flatten | 50 378 | 31 370 | 95.24 | 90.45 | 4.79 | 91.74 (8) |
| avg + GAP | 19 658 | 650 | 80.59 | 80.27 | 0.33 | 82.24 (8) |
| avg + Flatten | 50 378 | 31 370 | 94.98 | **92.06** | 2.92 | 92.16 (9) |
| stride + GAP | 66 026 | 650 | 93.21 | 89.82 | 3.39 | 89.82 (10) |
| stride + Flatten | 96 746 | 31 370 | 98.09 | 91.59 | 6.50 | 91.71 (6) |

Mapa przed głowicą to we wszystkich wariantach [64, 7, 7]: GAP podaje do warstwy `Linear` 64 cechy, a Flatten 3 136.

<img width="2684" height="667" alt="krzywe_B_glowica" src="https://github.com/user-attachments/assets/1c9b0f22-8653-4db9-9497-a87c8a017abf" />


### 3.4 Liczba bloków splot–redukcja

| Redukcja | Bloki | Parametry | Mapa przed GAP | Train acc [%] | Test acc [%] | Przerwa [p.p.] |
|---|---:|---:|---|---:|---:|---:|
| max | 1 | 714 | [32, 14, 14] | 64.05 | 63.89 | 0.16 |
| max | 2 | 19 658 | [64, 7, 7] | 84.56 | 83.60 | 0.96 |
| max | 3 | 94 410 | [128, 3, 3] | 93.16 | 89.34 | 3.82 |
| avg | 1 | 714 | [32, 14, 14] | 63.46 | 63.39 | 0.07 |
| avg | 2 | 19 658 | [64, 7, 7] | 80.59 | 80.27 | 0.33 |
| avg | 3 | 94 410 | [128, 3, 3] | 91.72 | 89.93 | 1.80 |
| stride | 1 | 10 026 | [32, 14, 14] | 78.98 | 79.00 | −0.02 |
| stride | 2 | 66 026 | [64, 7, 7] | 93.21 | 89.82 | 3.39 |
| stride | 3 | 288 618 | [128, **4, 4**] | 98.38 | **92.31** | 6.07 |

Przy trzech blokach wymiar 7 jest nieparzysty. Pooling zaokrągla w dół (⌊7/2⌋ = 3), a splot z krokiem 2 i paddingiem 1 w górę (⌊(7 + 2 − 3)/2⌋ + 1 = 4).

<img width="2684" height="667" alt="krzywe_C_bloki" src="https://github.com/user-attachments/assets/86c0c002-2a6a-4927-89c5-0eb5dd0756b9" />


## 4. Analiza i interpretacja

**Max vs avg vs stride (2 bloki, GAP).**
- Modele z poolingiem są niedouczone. Train acc wynosi tylko 80–85% i jest prawie równa test acc (przerwa 0.3–1.0 p.p.), więc ograniczeniem jest pojemność, a nie przeuczenie.
- Splot z krokiem podnosi test acc do 89.82% (+6.22 p.p.). Wynik ten ma dwie przyczyny: uczone wagi redukcji oraz to, że sieć ma 4 zamiast 2 warstw splotowych z nieliniowością. Kosztem jest 3.36× więcej parametrów i większa przerwa train–test (3.39 p.p.).
- Average pooling jest najsłabszy (80.27%). Przy GAP na końcu sieć uśrednia dwukrotnie: najpierw lokalnie w poolingu, potem globalnie w GAP. Słabe, lokalne aktywacje zostają rozmyte. Max pooling zachowuje najsilniejszą odpowiedź w oknie, co daje +3.33 p.p. bez dodatkowych parametrów.

**Okno 2×2 vs 3×3.**
- Dla wejścia 28 × 28 oba okna dają tę samą mapę [64, 7, 7] (28 → 14 → 7). Liczba parametrów też jest taka sama, bo pooling nie ma wag.
- Różnice po ostatniej epoce to −0.89 p.p. (max) i +1.42 p.p. (avg). Są mniejsze niż wahania między kolejnymi epokami: dla max 3×3 accuracy spada z 85.93% do 81.93% między epoką 6 a 7. Na podstawie jednego przebiegu nie da się więc stwierdzić wpływu rozmiaru okna.

**GAP vs Flatten.**
- Flatten zwiększa liczbę parametrów głowicy 48× (650 → 31 370) i podnosi test acc:
  - max: +6.85 p.p.,
  - avg: +11.79 p.p.,
  - stride: +1.77 p.p.
- Rośnie jednak przerwa train–test, odpowiednio do 4.79, 2.92 i 6.50 p.p. Głowica Flatten zachowuje położenie aktywacji w mapie 7 × 7, które GAP usuwa. Ta informacja pomaga przy słabej części splotowej, ale ułatwia też dopasowanie do zbioru treningowego.
- Największy zysk ma average pooling: bez GAP znika problem podwójnego uśredniania, a avg + Flatten daje najlepszy wynik wśród sieci dwublokowych (92.06%).
- Dla stride zysk jest najmniejszy, bo część splotowa jest już silna. Flatten głównie zwiększa przeuczenie (train acc 98.09%).

**Liczba bloków.**
- 1 blok: mapa [32, 14, 14], test acc 63–79%. Jedna warstwa splotowa i GAP to za mało, żeby rozróżnić klasy; max i avg dają praktycznie ten sam wynik (63.89% vs 63.39%).
- Każdy kolejny blok wyraźnie podnosi accuracy (max: 63.89 -> 83.60 -> 89.34%), a przy trzech blokach różnice między max i avg prawie znikają (89.34% vs 89.93%).
- Wraz z blokami rośnie też liczba kanałów (32 → 64 → 128). Zmienia się więc jednocześnie głębokość i szerokość, a efektu nie można przypisać samej głębokości.
- Najlepszy wynik ma stride z 3 blokami (92.31%), ale kosztem 288 618 parametrów i największego przeuczenia (train 98.38%, przerwa 6.07 p.p.).

**Kompromis jakość ↔ parametry.**

| Konfiguracja | Parametry | Test acc [%] |
|---|---:|---:|
| max + GAP | 19 658 | 83.60 |
| stride + GAP | 66 026 | 89.82 |
| max, 3 bloki | 94 410 | 89.34 |
| avg + Flatten | 50 378 | 92.06 |
| stride, 3 bloki | 288 618 | 92.31 |

avg + Flatten osiąga prawie ten sam wynik co stride z 3 blokami (−0.25 p.p.) przy 5.7× mniejszej liczbie parametrów. Uczona redukcja (stride) daje większy zysk na parametr niż dokładanie bloków z poolingiem: stride + GAP ma +0.48 p.p. przy 30% mniejszej liczbie parametrów niż max z 3 blokami.

**Ograniczenia.**
- Każdy wariant to jeden przebieg z jednym seedem.
- Test acc modeli z GAP waha się między epokami o 2–4 p.p.; np. avg 2×2 w epokach 5–8: 79.68 → 76.29 → 78.45 → 82.24%. Różnice poniżej ok. 3 p.p. (okno 2×2 vs 3×3, max vs avg przy 3 blokach) nie są rozstrzygające.
- 10 epok to za mało, żeby modele z GAP się zbiegły: krzywe max i avg 2×2 jeszcze rosną.

## 5. Wnioski

1. Przy jednakowym budżecie treningowym splot 3×3 z krokiem 2 dał najwyższą accuracy (89.82%), wyraźnie wyższą niż max (83.60%) i avg pooling (80.27%). Kosztem są 46 368 dodatkowych parametrów (×3.36) i dodatkowa głębokość, z których bierze się większa część zysku.
2. Max pooling jest lepszy od average poolingu w sieci z GAP (+3.33 p.p.) przy zerowym koszcie parametrów. Avg pooling dobrze wypada dopiero bez GAP (avg + Flatten: 92.06%).
3. Okno 3×3 zamiast 2×2 nie zmienia rozmiaru mapy dla parzystych wymiarów (28 -> 14 -> 7). Różnica w accuracy (≤ 1.4 p.p.) mieści się w szumie między epokami.
4. GAP zmniejsza głowicę 48× (650 vs 31 370 parametrów) i ogranicza przeuczenie. W małej, dwublokowej sieci kosztuje to jednak 1.8–11.8 p.p. accuracy, bo usuwa informację o położeniu cech.
5. Głębsza część splotowa (3 bloki) niweluje różnice między max i avg, a z GAP daje 89.3–92.3%. Przy stride przeuczenie jest jednak największe (przerwa 6.07 p.p.).
6. Kierunki poprawy:
   - powtórzenie przebiegów dla 3–5 seedów (średnia ± std),
   - dłuższy trening ze schedulerem lr dla modeli z GAP,
   - porównanie stride vs pooling przy wyrównanej głębokości (pooling poprzedzony dodatkowym splotem 3×3 z krokiem 1),
   - augmentacja lub dropout dla wariantów z Flatten.

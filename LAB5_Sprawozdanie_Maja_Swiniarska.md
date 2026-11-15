# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 5 — Warstwy splotowe: operacja, parametry, interpretacja

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 13.11.2026

## 1. Cel ćwiczenia

Zbadanie działania warstwy splotowej na własnym zdjęciu w skali szarości (512×384 px): ręczne jądra przez `F.conv2d`, weryfikacja wzoru na wymiary wyjścia względem `nn.Conv2d`, wpływ `stride`/`padding`/`dilation`, interpretacja map cech pretrenowanego ResNet-18 oraz porównanie filtrów rozmywających (Box 3×3, Box 5×5, Gaussian 5×5). Zakres dodatkowy: filtry płytkiej i głębokiej warstwy, własne jądro na krawędzie ukośne, liczba parametrów `Conv2d` vs warstwa gęsta.

**Zdjęcie przed filtracją** (to samo we wszystkich punktach):

<img width="549" height="756" alt="00_zdjecie_oryginalne" src="https://github.com/user-attachments/assets/e486d3dd-4a68-42eb-bab1-8c83ddbb7756" />


## 2. Metoda i implementacja

- Narzędzia: PyTorch 2.14.1 (CUDA), torchvision, matplotlib, pandas. Model: ResNet-18 (`ResNet18_Weights.DEFAULT`).
- Zdjęcie: konwersja do skali szarości (`"L"`), skalowanie do dłuższego boku 512 px (LANCZOS), tensor `[1,1,512,384]`, wartości 0–1.
- Splot: `F.conv2d`, `padding = k//2` (zachowanie rozmiaru dla `s=1, d=1`).
- Wzór na wymiar osi: `out = (in + 2p − d(k−1) − 1)//s + 1`.
- Sieć: wejście 3-kanałowe = ta sama skala szarości w każdym kanale (`pil_gray.convert("RGB")`), preprocessing `weights.transforms()` (resize 256, crop 224, normalizacja ImageNet); mapy z `conv1` po ReLU.
- [Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB5_Zadanie_Maja_Swiniarska.ipynb)

## 3. Wyniki

### 3.1 Ręczne jądra

<img width="1935" height="1766" alt="01_reczne_jadra" src="https://github.com/user-attachments/assets/7b4413a8-3ab5-476c-b5d6-057785f2c32b" />


Sobel X/Y pokazane ze znakiem (paleta RdBu, wspólna skala ±3,89), pozostałe na skali 0–1.

| Jądro | Wagi | Co robi | Wynik na zdjęciu |
|---|---|---|---|
| Sobel X | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | prawa kolumna − lewa; wykrywa krawędzie **pionowe** (okna, nadkola, słupki szyby) | zakres [−3,78; 3,46], śr. \|gx\| = 0,212 |
| Sobel Y | transpozycja Sobela X | górny − dolny wiersz; wykrywa krawędzie **poziome** (dach, zderzak, fugi bruku) | zakres [−3,89; 3,82], śr. \|gy\| = 0,344 |
| Magnituda | √(gx²+gy²) | siła zmiany jasności niezależnie od kierunku | maks. 3,92, śr. 0,468 |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]`, suma 1 | środek +5, sąsiedzi −1; zwiększa kontrast przy krawędziach | przed `clamp` zakres [−1,67; 2,59]; 5,9% pikseli <0 i 7,5% >1; odch. std. 0,234 → 0,307 |
| Box 5×5 | wagi 1/25, suma 1 | lokalna średnia; wygładza i osłabia drobny detal | odch. std. 0,234 → 0,213; śr. magnituda gradientu 0,468 → 0,219 (47%) |

Na tym zdjęciu poziome krawędzie są silniejsze niż pionowe (śr. \|gy\| / \|gx\| = 1,63).

### 3.2 Wzór na wymiary wyjścia 

Wejście 32×32×3, 8 filtrów (`nn.Conv2d(3, 8, ...)`).

| k | s | p | d | k_eff = 1+d(k−1) | licznik = 32+2p−d(k−1)−1 | wzór | `Conv2d` | zgodne |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 1 | 1 | 3 | 31 | 32 | 32 | ✓ |
| 3 | 2 | 1 | 1 | 3 | 31 | 16 | 16 | ✓ |
| 5 | 1 | 2 | 1 | 5 | 31 | 32 | 32 | ✓ |
| 3 | 1 | 2 | 2 | 5 | 31 | 32 | 32 | ✓ |
| 7 | 2 | 3 | 1 | 7 | 31 | 16 | 16 | ✓ |
| 3 | 1 | 0 | 1 | 3 | 29 | 30 | 30 | ✓ |
| 5 | 3 | 1 | 1 | 5 | 29 | 10 | 10 | ✓ |

7 z 7 konfiguracji zgodnych. Rozmiar zachowany dla `s=1` i `2p = d(k−1)` (wiersze 1, 3, 4).

### 3.3 Wpływ stride, padding i dilation 

Sobel X, wartość bezwzględna, wspólna skala jasności; wejście 512×384; czerwona ramka = rozmiar wejścia.

<img width="2106" height="1784" alt="02_stride_padding_dilation" src="https://github.com/user-attachments/assets/47d56d92-b49f-421b-944b-d8cb02b7b836" />


| Parametr | Ustawienie | Wyjście | Uwagi |
|---|---|---|---|
| odniesienie | s=1, p=1, d=1 | 512×384 | 196 608 wartości |
| stride | s=2, p=1, d=1 | 256×192 | 49 152 wartości (1/4) |
| stride | s=4, p=1, d=1 | 128×96 | 12 288 wartości (1/16) |
| padding | s=1, p=0, d=1 | 510×382 | −2 w każdej osi (brzeg odcięty) |
| padding | s=1, p=3, d=1 | 516×388 | +4 w każdej osi |
| dilation | s=1, p=2, d=2 | 512×384 | k_eff = 5; rozmiar zachowany dzięki p = d(k−1)/2 = 2 |

Dla każdego ustawienia rozmiar z `F.conv2d` = rozmiar ze wzoru (wartości w tytułach paneli).

### 3.4 Mapy cech pretrenowanej sieci 

Warstwa: `conv1` ResNet-18 (64 filtry 7×7×3 = 147 wag każdy, k=7, s=2, p=3) → 64 mapy 112×112 po ReLU.

<img width="1771" height="889" alt="03a_filtry_conv1" src="https://github.com/user-attachments/assets/ed725f83-a7f6-4bab-9174-dfff816710c3" />


<img width="2085" height="686" alt="03b_mapy_conv1" src="https://github.com/user-attachments/assets/ac0664d8-b838-45a4-aa1e-8e719b29b5ab" />

<img width="2535" height="772" alt="03c_filtry_i_mapy_top8" src="https://github.com/user-attachments/assets/be139873-3a64-437e-87ac-de88b50b0f2f" />


Na rysunku 3c filtr RGB sprowadzono do jednego filtra szarości (suma W_c/σ_c), bo kanały wejścia niosą ten sam sygnał.

| Filtr | Wzór wag | Co podświetla mapa |
|---|---|---|
| 26, 57 | poziomy podział: góra ujemna/dół dodatnia (26) i odwrotnie (57) | krawędzie **poziome** — linia dachu, zderzak, listwa boczna; przeciwna polaryzacja |
| 14, 25 | pionowy podział: lewa strona ujemna/prawa dodatnia (14) i odwrotnie (25) | krawędzie **pionowe** — okna budynku, nadkola, słupki szyby |
| 10 | wąski poziomy pasek wag dodatnich | cienkie poziome linie (fugi bruku, listwy) |
| 19 | ukośny podział wag | ukośne krawędzie, łuki kół |
| 32, 61 | wagi jednego znaku (plama) | wygładzony obraz jasności; 32 ma największą średnią odpowiedź |

Z 16 pierwszych map (rys. 3b): mapy 0 i 8 — drobne prążki bruku (wysokie częstotliwości, odpowiednio poziome i pionowe); mapy 2, 4, 6, 9 — wygładzona jasność; mapy 3, 7, 13 — ciemne obszary (opony, wnęki kół, cień pod autem) świecą na ciemnym tle karoserii; mapy 1, 5, 10, 11, 14 — krawędzie (obrys auta, okna). ReLU zeruje odpowiedzi ujemne, stąd ciemne tło wielu map. Mapy obejmują środkowy kwadrat 224×224 (center crop), dlatego auto jest przycięte po bokach.

### 3.5 Porównanie filtrów rozmywających 

Filtry: Box 3×3 (wagi 1/9, `padding=1`), Box 5×5 (wagi 1/25, `padding=2`), Gaussian 5×5 (`[1,4,6,4,1]⊗[1,4,6,4,1]/256`, `padding=2`); `stride=1`, `dilation=1`; wyjście 512×384 dla każdego, suma wag = 1. Fragment: wiersze 250–345, kolumny 270–365 (grill, reflektor, tablica rejestracyjna), wspólna skala jasności 0–1.

<img width="549" height="754" alt="04a_fragment" src="https://github.com/user-attachments/assets/16a78b01-6b25-454b-a819-ca8bd930d0a1" />


<img width="2685" height="708" alt="04b_fragment_przed_po" src="https://github.com/user-attachments/assets/4524946e-c74d-4344-9b8a-737e89157781" />


<img width="1270" height="587" alt="04c_profil_jasnosci" src="https://github.com/user-attachments/assets/59effa1c-4da9-4671-ad62-6f4b2f27a303" />


<img width="2002" height="683" alt="04d_wagi_jader" src="https://github.com/user-attachments/assets/5d36327f-2e0d-41e9-abe1-6108f9074d94" />


**Siła krawędzi we fragmencie** (magnituda Sobela):

| Filtr | śr. \|∇\| | % oryginału | maks. \|∇\| | odch. std. fragmentu | rozpiętość jasności znaków tablicy (wiersz 312, kol. 320–357) | rozpiętość jasności szczelina–listwa grilla (kol. 330, wiersze 262–295) |
|---|---|---|---|---|---|---|
| przed filtracją | 0,779 | 100,0% | 3,92 | 0,291 | 0,65 | 0,85 |
| Box 3×3 | 0,492 | 63,1% | 2,18 | 0,261 | 0,27 | 0,55 |
| Box 5×5 | 0,292 | 37,5% | 1,28 | 0,244 | 0,22 | 0,48 |
| Gaussian 5×5 | 0,434 | 55,7% | 1,98 | 0,255 | 0,25 | 0,52 |

Szerokość przejścia jasność 0,5 → 0,15 na krawędzi tablica–opona (wiersz 312, kol. 356–360): oryginał 2 px, Box 3×3 3 px, Gaussian 5×5 3 px, Box 5×5 4 px.

**Rozkład wag jąder:**

| Jądro | waga środka | udział wag w zewn. pierścieniu | σ rozkładu wag (px na oś) |
|---|---|---|---|
| Box 3×3 | 0,1111 | 88,9% | 0,82 |
| Box 5×5 | 0,0400 | 64,0% | 1,41 |
| Gaussian 5×5 | 0,1406 | 23,4% | 1,00 |

### 3.6 Zakres dodatkowy

**Filtry płytkiej i głębokiej warstwy.**

<img width="2535" height="1297" alt="05a_mapy_plytka_vs_gleboka" src="https://github.com/user-attachments/assets/a9461100-b3a8-46b2-8b2a-1f0d53b00983" />


<img width="1935" height="559" alt="05b_filtr_gleboki" src="https://github.com/user-attachments/assets/8f2e1787-8b40-4f05-98dd-03ee338a9b96" />


| Warstwa | Kształt wag | Wag na filtr | Mapy |
|---|---|---|---|
| `conv1` | (64, 3, 7, 7) | 147 | 64 × 112×112 |
| `layer1.0.conv1` | (64, 64, 3, 3) | 576 | 64 × 56×56 |
| `layer3.0.conv1` | (256, 128, 3, 3) | 1152 | 256 × 14×14 |

**Własne jądro — krawędzie ukośne.** `diag_a = [[0,1,2],[-1,0,1],[-2,-1,0]]` (krawędzie `\`), `diag_b` = odbicie kolumn (krawędzie `/`).

<img width="1935" height="1685" alt="06_wlasne_jadro_ukosne" src="https://github.com/user-attachments/assets/943449d3-cff1-48e3-af97-2449bac0bc52" />


Na rombie testowym `diag_a` daje odpowiedź tylko na dwóch bokach biegnących po przekątnej `\` (północno-wschodni ujemny, południowo-zachodni dodatni), a `diag_b` tylko na dwóch pozostałych (`/`); na bokach o przeciwnej orientacji odpowiedź jest zerowa.

**Liczba parametrów `Conv2d` vs warstwa gęsta o tym samym wejściu i wyjściu:**

| Przypadek | Wejście → wyjście | `Conv2d` | Gęsta | Stosunek |
|---|---|---|---|---|
| `conv1` ResNet-18, k=7, bez biasu | 3×224×224 -> 64×112×112 | 9 408 | 120 846 286 848 | 12 845 056× |
| Zdjęcie, 1→8 filtrów, k=3 | 1×512×384 -> 8×512×384 | 80 | 309 239 218 176 | 3 865 490 227× |
| 32×32 RGB, 16 filtrów, k=3 | 3×32×32 -> 16×32×32 | 448 | 50 348 032 | 112 384× |

Wzory: `Conv2d` = C_out·C_in·k² (+ C_out biasów); gęsta = (C_in·H·W)·(C_out·H_out·W_out) (+ biasy). Wartości `Conv2d` zgodne z liczbą parametrów z PyTorch (9 408 / 80 / 448).

## 4. Analiza i interpretacja

**Ręczne jądra.** Sobel X i Y rozdzielają krawędzie według orientacji (Y silniejszy: 0,344 vs 0,212, bo zdjęcie zawiera dużo poziomych linii: fugi bruku, dach, zderzak). Magnituda łączy obie orientacje. Sharpen podnosi odch. std. z 0,234 do 0,307, ale 13,4% pikseli wychodzi poza 0–1, więc wymaga `clamp`. Box 5×5 zmniejsza średnią magnitudę gradientu o 53%.

**Wymiary.** Wzór zgadza się z `nn.Conv2d` we wszystkich 7 przypadkach. Zaokrąglenie w dół widać dla `k=5, s=3, p=1`: licznik 29, 29//3 = 9, wynik 10.

**Parametry splotu.** `stride` skraca mapę (÷s w każdej osi: 512×384 -> 256×192 → 128×96). `padding` decyduje o brzegu (p=0: −2 px, p=3: +4 px). `dilation=2` przy `k=3` daje okno 5×5 przy tych samych 9 wagach; krawędzie w mapie są szersze, a rozmiar zachowuje `p=2`.

**Mapy cech.** Filtry `conv1` dzielą się na kierunkowe detektory krawędzi (poziome 26/57/10, pionowe 14/25, ukośny 19) i filtry „plamowe" o wagach jednego znaku (32, 61), które działają jak wygładzenie. Kierunkowe filtry mają ten sam charakter co ręczne Sobel Y i Sobel X z pkt 3.1. Mapy 3, 7, 13 pokazują, że część filtrów reaguje na ciemne obszary. W głębszych warstwach mapy tracą związek z obrazem: `layer1.0.conv1` (56×56) daje zgrubne, blokowe odpowiedzi na części auta (koła, zderzak, dach), a `layer3.0.conv1` (14×14, 128 kanałów wejścia) nie przypomina już zdjęcia i ma pojedyncze jasne komórki. Pojedynczych filtrów 3×3 z głębszej warstwy (rys. 5b) nie da się zinterpretować wizualnie, bo każdy sumuje wkłady 64 kanałów (576 wag).

**Box 3×3 vs Box 5×5.** Box 5×5 uśrednia 25 pikseli zamiast 9 (σ wag 1,41 vs 0,82 px), więc usuwa więcej detalu: średnia magnituda gradientu we fragmencie spada do 37,5% oryginału wobec 63,1% dla Box 3×3, a maksymalna do 1,28 wobec 2,18. Konkretnie: po Box 3×3 szczeliny grilla pozostają rozdzielone (rozpiętość jasności szczelina–listwa 0,55), po Box 5×5 zlewają się w rozmytą kratkę o niskim kontraście (0,48); cyfry tablicy są nieczytelne już po Box 3×3 (rozpiętość 0,65 → 0,27), po Box 5×5 ciemne plamy znaków zlewają się dalej (0,22). Krawędź tablica–opona jest rozmyta na 3 px (Box 3×3) vs 4 px (Box 5×5), przy 2 px w oryginale.

**Box 5×5 vs Gaussian 5×5.** Mimo tego samego okna Gaussian zachowuje więcej detalu: 55,7% vs 37,5% średniej magnitudy gradientu, maks. 1,98 vs 1,28, kontrast grilla 0,52 vs 0,48, krawędź tablica–opona rozmyta na 3 px vs 4 px. Wynika to z rozkładu wag: Box nadaje każdemu z 25 pikseli 0,04 (zewnętrzny pierścień niesie 64% wag, σ = 1,41 px), a Gaussian skupia wagę w środku (0,1406, narożniki 0,004), zewnętrzny pierścień niesie 23,4%, σ = 1,00 px. Efektywny zasięg rozmycia Gaussiana 5×5 (σ = 1,00) jest bliższy Box 3×3 (σ = 0,82) niż Box 5×5, co zgadza się z wynikami (55,7% vs 63,1% vs 37,5%).

## 5. Wnioski

1. Wzór `out = (in + 2p − d(k−1) − 1)//s + 1` dał poprawny wymiar we wszystkich 7 sprawdzonych konfiguracjach; przy `s=1` rozmiar zachowuje `2p = d(k−1)`.
2. `stride` zmniejsza liczbę wartości mapy ≈ s² razy (196 608 -> 49 152 -> 12 288), `padding` kontroluje brzeg (±2 px na oś przy p=0/p=3), `dilation` zwiększa zasięg okna (k_eff = 5 przy k=3, d=2) bez dodatkowych wag.
3. Filtry `conv1` ResNet-18 odpowiadają ręcznym jądrom: kierunkowe detektory krawędzi (poziome, pionowe, ukośne) i filtry wygładzające; głębsze warstwy dają mniejsze (56×56, 14×14) mapy, których nie da się odnieść do pojedynczych krawędzi.
4. Rozmycie Box 5×5 usuwa więcej detalu niż Box 3×3 (37,5% vs 63,1% gradientu oryginału); Gaussian 5×5 przy tym samym oknie zachowuje 55,7%, bo większość wagi leży w środku okna.
5. Warstwa splotowa ma o 5–10 rzędów wielkości mniej parametrów niż warstwa gęsta o tym samym wejściu i wyjściu (9 408 vs ~1,2·10¹¹ dla `conv1` ResNet-18), dzięki lokalności i współdzieleniu wag.
6. Ograniczenia: wyniki z jednego zdjęcia i jednego fragmentu; dopełnienie zerami przyciemnia brzeg obrazu (fragment wybrano z dala od brzegu); mapy sieci oglądane po ReLU bez BatchNorm, na przyciętym środku kadru 224×224.

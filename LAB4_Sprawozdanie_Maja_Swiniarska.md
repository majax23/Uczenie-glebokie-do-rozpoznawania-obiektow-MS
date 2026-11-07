# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 04. Splotowe sieci neuronowe (CNN)

**Autor / Grupa:** Maja Świniarska GL.04

**Data:** 04.11.2026

## 1. Cel ćwiczenia

Porównanie sieci MLP (baseline) z prostą siecią splotową SimpleCNN na CIFAR-10, wytrenowanych w identycznych warunkach. Porównywane wielkości: accuracy (walidacja i test), liczba parametrów, przeuczenie, czas uczenia.
Zakres dodatkowy: mapy cech dwóch warstw splotowych, wpływ liczby filtrów (16/32, 32/64, 64/128) oraz liczby bloków (1, 2, 3).

## 2. Metoda i implementacja

**Środowisko:** Google Colab, GPU Tesla T4, PyTorch 2.11.0+cu130, torchvision, matplotlib, pandas.

[Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB4_Zadanie_Maja_Swiniarska.ipynb)

**Dane:** CIFAR-10 (60 000 obrazów 3×32×32, 10 klas), wczytane z archiwum z Dysku Google. Podział: 45 000 trening / 5 000 walidacja (losowany ze stałym ziarnem 42, identyczny dla obu modeli) / 10 000 test. Normalizacja per kanał (średnia i odchylenie zbioru treningowego). Bez augmentacji.

**Modele:**

| Model | Architektura |
|---|---|
| MLP | Flatten -> Linear(3072, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 10) |
| SimpleCNN | 2 bloki: [Conv3×3(pad 1) -> BatchNorm -> ReLU] ×2 -> MaxPool2, filtry 32 i 64; głowica: Flatten -> Linear(4096, 256) -> ReLU -> Dropout(0,3) -> Linear(256, 10). Kształty: 3×32×32 -> 32×16×16 -> 64×8×8 |

**Trening (wspólny dla obu modeli):** Adam, lr = 1e-3, batch_size = 128, 10 epok, CrossEntropyLoss (surowe logity), seed 42. Po każdej epoce liczona strata i accuracy na treningu (tryb `eval`, bez dropoutu) oraz walidacji. Wybierana epoka o najwyższej accuracy walidacyjnej, wagi z tej epoki przywracane. Zbiór testowy użyty jeden raz na model, po wyborze epoki. Czas uczenia = same pętle treningowe (z `torch.cuda.synchronize()`), ewaluacja liczona osobno. Wszystkie czasy zmierzono w jednej sesji Colab.

## 3. Wyniki

### Testy kontrolne przed treningiem

| Test | Wynik | Oczekiwane |
|---|---|---|
| Kształt wyjścia SimpleCNN dla N=4 | (4, 10) | (4, 10) |
| Strata na starcie — MLP | 2,309 | ok. 2,30 |
| Strata na starcie — CNN | 2,303 | ok. 2,30 |
| Przeuczenie jednej paczki (32 obrazy): strata w kroku 0 → 100 | 2,374 → 0,0001 | spadek bliski 0 |


### Porównanie MLP i SimpleCNN

**Tabela 1.** Wyniki (epoka wybrana na walidacji; accuracy treningowa w trybie `eval`, bez augmentacji).

| Wskaźnik | MLP | SimpleCNN |
|---|---|---|
| Parametry — łącznie | 1 707 274 | 1 117 354 |
| Parametry — część splotowa (features) | — | 65 952 |
| Parametry — głowica gęsta (classifier) | — | 1 051 402 |
| Epoka wybrana na walidacji | 8 | 10 |
| Accuracy trening [%] | 70,59 | 90,54 |
| Accuracy walidacja [%] | 53,54 | 80,40 |
| Przerwa train–walidacja [p.p.] | 17,05 | 10,14 |
| Strata walidacyjna | 1,464 | 0,610 |
| **Accuracy test [%]** | **52,03** | **79,83** |
| Strata testowa | 1,465 | 0,618 |
| Czas uczenia [s] | 103,7 | 115,8 |
| Czas ewaluacji [s] | 106,4 | 114,0 |

Klasyfikator losowy: 10%. Nad losowym: MLP +42,03 p.p., CNN +69,83 p.p.
Czas ewaluacji obejmuje w każdej epoce przejście po 45 000 obrazach treningowych i 5 000 walidacyjnych.


**Rys. 1.** Krzywe uczenia: strata (lewo) i accuracy (prawo), trening (linia ciągła) i walidacja (przerywana); pionowe linie = wybrana epoka.

<img width="2580" height="893" alt="04_krzywe_uczenia" src="https://github.com/user-attachments/assets/d822615f-f0d7-4e5a-9b21-e0f346337e0f" />


**Rys. 2.** Przerwa train–walidacja w kolejnych epokach (accuracy i strata).

<img width="2181" height="756" alt="05_przerwa_train_walidacja" src="https://github.com/user-attachments/assets/d103a4e2-83a4-4c55-88fa-1d2bbf6a6877" />


Przerwa w accuracy rośnie w obu modelach, ale w MLP wyraźnie szybciej: epoka 10 — MLP 22,07 p.p., CNN 10,14 p.p. (pełna tabela per epoka: `04_historia_uczenia.csv`).

###  Wyniki na klasach (zbiór testowy)

**Rys. 3.** Accuracy testowa na klasę.

<img width="2180" height="780" alt="07_accuracy_na_klase" src="https://github.com/user-attachments/assets/ff3568b5-cb87-4db2-a3dd-bf9e71df9a87" />


**Rys. 4.** Macierze pomyłek (znormalizowane po wierszach, %).

<img width="2512" height="1179" alt="08_macierze_pomylek" src="https://github.com/user-attachments/assets/61197412-f1ef-4881-92a7-8b6d54dae703" />


Najlepsze klasy CNN: ship 92,4%, truck 87,6%, automobile 86,6%. Najgorsze: dog 66,8%, bird 67,6%. Najlepsze klasy MLP: ship 67,5%, automobile 64,5%; najgorsze: bird 34,0%, dog 42,1%, cat 42,2%, deer 42,7%. Największy wzrost dla CNN: deer +37,4 p.p., bird +33,6 p.p.; najmniejszy: horse +16,6 p.p.

### Mapy cech (zakres dodatkowy)

**Rys. 5.** Mapy cech (po ReLU, bez BatchNorm) dla obrazu testowego „cat”: góra — `features[0][0]` (32×32), dół — `features[1][0]` (16×16), pierwsze 8 filtrów.

<img width="2979" height="777" alt="09_mapy_cech" src="https://github.com/user-attachments/assets/0bf1a8be-ccac-4c90-84b1-158cefe4f104" />


### Wpływ liczby filtrów i liczby bloków (zakres dodatkowy, walidacja)

Konfiguracja bazowa (32/64, 2 bloki) pochodzi z głównego treningu. Pozostałe: ta sama konfiguracja `CFG`. Test nie był używany.

**Tabela 2.** Liczba filtrów w blokach.

| Konfiguracja | Parametry łącznie | Splotowa | Głowica | Epoka | Acc walidacja [%] | Przerwa [p.p.] | Czas [s] |
|---|---|---|---|---|---|---|---|
| 16/32 | 543 962 | 16 848 | 527 114 | 6 | 76,84 | 5,85 | 120,5 |
| 32/64 (bazowa) | 1 117 354 | 65 952 | 1 051 402 | 10 | 80,40 | 10,14 | 115,8 |
| 64/128 | 2 360 906 | 260 928 | 2 099 978 | 10 | 80,62 | 7,74 | 135,2 |

<img width="1418" height="469" alt="image" src="https://github.com/user-attachments/assets/b655febb-d5c4-4011-ac8d-146c4123532b" />


**Tabela 3.** Liczba bloków splotowych.

| Konfiguracja | Parametry łącznie | Splotowa | Głowica | Epoka | Acc walidacja [%] | Przerwa [p.p.] | Czas [s] |
|---|---|---|---|---|---|---|---|
| 1 blok (32) | 2 110 250 | 10 272 | 2 099 978 | 10 | 71,44 | 14,34 | 118,8 |
| 2 bloki (32/64) (bazowa) | 1 117 354 | 65 952 | 1 051 402 | 10 | 80,40 | 10,14 | 115,8 |
| 3 bloki (32/64/128) | 815 018 | 287 904 | 527 114 | 10 | 82,96 | 11,12 | 128,6 |

<img width="1418" height="469" alt="image" src="https://github.com/user-attachments/assets/9c96c5ed-c718-436e-b2ac-b127e795e65d" />


## 4. Analiza i interpretacja

**Różnica MLP vs CNN.** SimpleCNN osiągnął 79,83% na teście wobec 52,03% dla MLP (+27,8 p.p.). Przy 10 000 próbek testowych niepewność pojedynczego wyniku to ok. ±1 p.p., więc różnica jest wielokrotnie większa od szumu pomiarowego. CNN wygrywa w każdej z 10 klas (+16,6 … +37,4 p.p.); jego najsłabsza klasa (dog, 66,8%) jest na poziomie najlepszej klasy MLP (ship, 67,5%).

**Liczba parametrów.** CNN ma 35% mniej parametrów niż MLP (1,12 mln vs 1,71 mln), a daje wyraźnie lepszy wynik. W MLP 92% parametrów to pierwsza warstwa (3072·512). W CNN część splotowa to tylko 65 952 parametry (5,9%), a 94,1% wag leży w jednej warstwie Linear(4096, 256). Eksperyment z filtrami to potwierdza: 64/128 ma 2,1× więcej parametrów niż 32/64, a accuracy walidacyjna jest praktycznie taka sama (80,62% vs 80,40%; różnica 0,2 p.p. mieści się w szumie ok. ±1,1 p.p. dla 5 000 próbek walidacyjnych). Wzrost liczby parametrów nie jest więc źródłem przewagi CNN.

**Własności splotu.** Dane z notatnika wskazują na dwie własności:
- *Współdzielenie wag:* ten sam filtr jest stosowany w każdym miejscu obrazu, więc liczba parametrów warstwy nie zależy od rozmiaru obrazu. Conv2d(3, 32, 3) ma 32·(3·3·3+1) = 896 parametrów, a cała część splotowa SimpleCNN 65 952. Pierwsza warstwa MLP (Linear(3072, 512)) ma 1 573 376 parametrów, czyli ok. 24× więcej niż cała część splotowa CNN, a wzorzec w innym miejscu obrazu trafia w niej na inne wagi.
- *Lokalność:* mapy pierwszej warstwy (Rys. 5) pokazują, że filtry 3×3 reagują na lokalne krawędzie i tekstury. Pole recepcyjne rośnie dopiero z głębokością (obliczone dla bloków conv-conv-pool: 1 blok 6×6, 2 bloki 16×16, 3 bloki 36×36 px, czyli cały obraz 32×32). W eksperymencie z blokami accuracy walidacyjna rośnie razem z nim: 71,44% -> 80,40% -> 82,96%.

Nie wynika stąd, że to sam splot daje całą przewagę: SimpleCNN ma też BatchNorm i Dropout, a MLP nie, i eksperyment tych czynników nie rozdziela.

**Przeuczenie.**
- *MLP:* mocne przeuczenie. Strata treningowa spada do 0,697, a walidacyjna po minimum w epoce 5 (1,372) rośnie do 1,569. Accuracy walidacyjna stoi w okolicy 52–53,5% od epoki 5, a treningowa rośnie do 75,4%. Przerwa rośnie z 2,7 p.p. (epoka 1) do 22,1 p.p. (epoka 10). Wybrana epoka 8 jest już po minimum straty walidacyjnej.
- *SimpleCNN:* przeuczenie umiarkowane, ale rosnące: przerwa z 0,65 p.p. do 10,14 p.p., strata treningowa 0,272 vs walidacyjna 0,610. Wybrana epoka to ostatnia (10), czyli budżet 10 epok jest prawdopodobnie za mały i wynik mógłby jeszcze wzrosnąć, ale rosnąca przerwa sugeruje, że pomogłaby też augmentacja.
- *Niestabilność:* w epoce 8 CNN ma skok straty (treningowa 0,416 -> 0,614, accuracy walidacyjna 78,66% -> 73,04%), po czym wraca do trendu w epoce 9.

**Przypadki błędne.** Wspólny wzorzec w obu modelach to pomyłki między podobnymi klasami zwierząt: dog→cat 29% (MLP) i 23% (CNN), cat→dog 23% (MLP) i 10% (CNN). CNN prawie wyeliminował pomyłki wśród pojazdów (automobile↔truck: 15%/18% -> 6%/5%; airplane→ship: 12% -> 6%), natomiast pomyłka dog -> cat prawie się nie zmieniła. MLP szczególnie słabo rozpoznaje klasę bird (34%), rozpraszając ją na cat, deer i dog.

**Mapy cech.** W pierwszej warstwie mapy zachowują strukturę obrazu 32×32: filtry 0, 2, 4, 6 reagują głównie na krawędzie (kontur głowy i uszu kota), filtr 5 na teksturę (sierść) w całym obrazie, filtr 1 na duże jasne obszary. Jasny pasek przy górnej krawędzi w filtrze 7 to prawdopodobnie efekt dopełnienia zerami. W głębszej warstwie mapy 16×16 są rzadkie i grube — pojedyncze aktywne punkty, kontur kota nie jest już rozpoznawalny, filtry 0 i 3 są prawie nieaktywne dla tego obrazu. Jest to zgodne z tym, że głębsze filtry reagują na większe fragmenty (większe pole recepcyjne), ale na podstawie jednego obrazu nie da się przypisać im konkretnych znaczeń.

**Czas uczenia.** MLP 103,7 s, CNN 115,8 s (×1,12), mimo że CNN wykonuje znacznie więcej obliczeń. Prawdopodobnie czas ograniczają dane (DataLoader, `num_workers=2`, transformacje na CPU), a nie GPU. W eksperymencie z filtrami czas nie rośnie monotonicznie (16/32: 120,5 s; 32/64: 115,8 s), co wskazuje na rozrzut rzędu kilku–kilkunastu sekund; zauważalnie dłużej trwa dopiero 64/128 (135,2 s) i 3 bloki (128,6 s).

**Ograniczenia.**
- Jeden przebieg (jedno ziarno) na konfigurację — różnice poniżej ok. 1 p.p. nie są interpretowalne (dotyczy m.in. 32/64 vs 64/128).
- Epoka wybrana na walidacji 5 000 próbek; w przypadku CNN i większości wariantów to ostatnia epoka, więc wyniki dotyczą budżetu 10 epok, nie zbieżności.
- W eksperymencie z blokami zmiana liczby bloków zmienia też rozmiar głowicy (2,10 mln → 1,05 mln → 0,53 mln parametrów), więc nie jest to w pełni „jeden czynnik”. Co ciekawe, w tym eksperymencie model z najmniejszą liczbą parametrów (3 bloki, 0,82 mln) ma najlepszy wynik, ale różnica względem 2 bloków (+2,56 p.p.) jest zaledwie ok. 2× większa od szumu walidacji.


## 5. Wnioski

1. SimpleCNN jest wyraźnie lepszy od MLP: 79,83% vs 52,03% accuracy testowej (+27,8 p.p.), przy 35% mniejszej liczbie parametrów i podobnym czasie uczenia (×1,12).
2. Przewaga nie wynika z liczby parametrów: w CNN 94% wag leży w głowicy gęstej, a podwojenie liczby filtrów (64/128) nie poprawia wyniku (80,62% vs 80,40%).
3. MLP przeucza się silnie (przerwa 22 p.p. w epoce 10, strata walidacyjna rośnie od epoki 5); CNN przeucza się łagodniej (10 p.p.), ale przerwa rośnie.
4. Głębokość ma większy wpływ niż szerokość: 1 blok 71,44% -> 2 bloki 80,40% -> 3 bloki 82,96% (walidacja; pole recepcyjne 6×6 -> 16×16 → 36×36 px), przy zmniejszającej się liczbie parametrów. Podwojenie filtrów dało tylko +0,2 p.p.
5. Wynik CNN nie jest zbieżny w 10 epokach (najlepsza epoka = ostatnia) i wykazuje niestabilność w epoce 8.

**Kierunki poprawy:** więcej epok z harmonogramem lr (stabilizacja), augmentacja (losowy crop + odbicie) i `weight_decay` przeciw rosnącej przerwie, mniejsza głowica, kilka ziaren na konfigurację dla oceny rozrzutu, osobny test wpływu BatchNorm i Dropout w MLP, aby wyodrębnić wkład samego splotu.

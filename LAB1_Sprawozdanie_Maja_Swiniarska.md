## Sprawozdanie 

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 1. Obraz jako dane — pierwszy klasyfikator obrazów

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 14.10.2026

## 1. Cel ćwiczenia

Celem ćwiczenia było zbudowanie pierwszego klasyfikatora obrazów przy użyciu klasycznego perceptronu wielowarstwowego (MLP). Zadanie polegało na wytrenowaniu modeli z różną liczbą warstw ukrytych, zbadaniu wpływu architektury oraz współczynnika uczenia na zdolności generalizacyjne sieci, a także analizie i interpretacji najczęściej popełnianych przez model błędów.

## 2. Metoda i implementacja

- **Zbiór danych:** Wykorzystano zbiór FashionMNIST* (obrazy ubrań w skali szarości, rozdzielczość 28x28 pikseli). Został on wybrany zamiast CIFAR-10 ze względu na ograniczenia sieci gęstych (MLP). Sieci tego typu wymagają spłaszczenia obrazu do wektora jednowymiarowego. Dla FashionMNIST wektor wejściowy ma rozmiar 784, natomiast dla CIFAR-10 wynosiłby 3072, co przy sieciach typu Fully Connected drastycznie zawyżyłoby liczbę parametrów i prowokowało silne zjawisko przeuczenia.
- **Narzędzia:** PyTorch (konstrukcja i uczenie modelu), torchvision (ładowanie danych, transformacje), matplotlib (wizualizacja krzywych i siatki predykcji), scikit-learn (macierz pomyłek). Trening z użyciem akceleracji GPU (CUDA).
- **Kod:** [Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB1_Zadanie_Maja_Swiniarska.ipynb)
- **Konfiguracja eksperymentu:** 
  - Optymalizator: Adam.
  - Funkcja straty: CrossEntropyLoss.
  - Testowane warianty architektury (warstwy ukryte): (128,), (256, 128), (512, 256, 128).
  - Badane współczynniki uczenia (Learning Rate): 0.01, 0.001, 0.0001.
  - Liczba epok: 8.

## 3. Wyniki

| Architektura warstw ukrytych | Liczba parametrów | Accuracy (Test) |
| :--- | :--- | :--- |
| `(128,)` | 101 770 | 88.18% |
| `(256, 128)` (Baseline) | 235 146 | 88.02% |
| `(512, 256, 128)` | 567 434 | 88.43% |

- **Porównanie z baseline i analiza architektur:** Jako punkt odniesienia przyjęto domyślny model posiadający dwie warstwy ukryte o rozmiarach `(256, 128)`. Osiągnął on dokładność na zbiorze testowym 88.02% przy ponad 235 tys. parametrów.
- Zmiana architektury na mniejszą, jednowarstwową `(128,)` nieznacznie poprawiła wynik (88.18%), jednocześnie redukując liczbę parametrów o ponad połowę (ok. 101 tys.). 
- Wariantem o najlepszym balansie między złożonością a dokładnością okazała się sieć `(128,)`. Choć najwyższą bezwzględną dokładność (88.43%) uzyskał model najgłębszy `(512, 256, 128)`, to przewaga ok. 0.4 punktu procentowego nad bazowym modelem nie uzasadnia drastycznego wzrostu liczby parametrów (ponad 567 tys.). Dowodzi to, że dla tak prostego zadania dokładanie kolejnych warstw nie przynosi już wymiernych korzyści.
- **Wpływ współczynnika uczenia:** Zbadano wpływ LR na wybranym optymalnym wariancie.
  - **LR = 0.01:** Funkcja straty spadała gwałtownie, ale szybko zaczynała oscylować, przez co model miał problem z optymalnym zbiegnięciem do minimum. Skutkowało to zauważalnie niższą dokładnością.
  - **LR = 0.001:** Wartość optymalna. Krzywe straty (treningowa i testowa) stabilnie i gładko opadały, dając najlepszy kompromis między czasem uczenia a wynikiem końcowym.
  - **LR = 0.0001:** Współczynnik zbyt niski. Funkcja straty obniżała się bardzo powoli, przez co po 8 epokach model był wciąż niedouczony.

## 4. Analiza i interpretacja

- **Ograniczenia modelu:** Zastosowanie perceptronu wielowarstwowego (MLP) wymusza spłaszczenie dwuwymiarowego obrazu (28x28) do jednowymiarowego wektora (784 elementy). W wyniku operacji Flatten sieć traci informacje o relacjach przestrzennych pomiędzy pikselami. Z tego względu osiągnięcie skuteczności wyraźnie wyższej niż ok. 88-89% za pomocą MLP na tym zbiorze jest bardzo trudne, a zwiększanie liczby parametrów przestaje przynosić efekty.
- **Przypadki błędne:** Brak zachowania struktury przestrzennej sprawia, że model świetnie radzi sobie z unikalnymi kształtami, ale często myli przedmioty o zbliżonym, globalnym rozkładzie pikseli i podobnej sylwetce. Z analizy wygenerowanej macierzy pomyłek wynika, że model miał największe problemy z poprawnym rozróżnieniem klas należących do górnej części garderoby: Shirt, T-shirt/top oraz Pullover. Zauważalna była również duża liczba pomyłek (87) pomiędzy klasami Coat a Pullover. Z kolei w kategorii obuwia model miał trudności z odróżnieniem Sneaker od Ankle boot.
- **Interpretacja krzywych uczenia:** Na wykresie funkcji straty dla architektury `(512, 256, 128)` można zaobserwować początki zjawiska przeuczenia. Strata na zbiorze treningowym stale i wyraźnie malała przez wszystkie epoki, natomiast strata na zbiorze testowym w okolicach 2-3 epoki uległa wypłaszczeniu, a następnie zaczęła nieznacznie rosnąć.
- Na wykresie obrazującym wpływ współczynnika uczenia widać, że optymalną wartością jest LR = 0.001, gdzie strata testowa obniża się płynnie i stosunkowo szybko. Zbyt wysoki współczynnik (LR = 0.01) spowodował gwałtowne oscylacje funkcji straty, a zbyt niski (LR = 0.0001) sprawił, że po 7 epokach model był wciąż silnie niedouczony.
  
## 5. Wnioski

**Podsumowanie:**
Ćwiczenie udowodniło, że sieci oparte na perceptronie wielowarstwowym (MLP) stanowią skuteczny punkt wyjścia do klasyfikacji obrazów ubrań, osiągając dobrą dokładność. Eksperymenty wykazały jednak, że zwiększanie głębokości sieci MLP drastycznie zwiększa liczbę parametrów modelu, co tylko w niewielkim stopniu poprawia skuteczność, a jednocześnie pogłębia ryzyko przeuczenia - model zaczyna zapamiętywać zbiór treningowy zamiast generalizować, co uwidacznia się w braku spadku straty testowej przy spadającej stracie treningowej. Prawidłowe działanie optymalizatora Adam jest niezbędne do osiągnięcia optymalnego minimum funkcji straty.

**Kierunki poprawy:**
Głównym ograniczeniem perceptronu wielowarstwowego przy analizie obrazów jest wymóg użycia operacji `Flatten`, która niszczy dwuwymiarową strukturę danych. Ponieważ sąsiednie piksele na obrazie trafiają w wektorze na odległe pozycje, kształt analizowanych obiektów przestaje być jawnie widoczny, co tłumaczy pomyłki modelu pomiędzy podobnymi klasami.

W celu poprawy wyników klasyfikacji można zastosować:
1.  **Splotowe sieci neuronowe (CNN):** Wymiana struktury MLP na architekturę konwolucyjną pozwoli na analizę dwuwymiarowego obrazu "okienkiem", które zachowa strukturę przestrzenną przy jednoczesnym zmniejszeniu liczby potrzebnych parametrów na warstwę.
2.  **Wykorzystanie zbioru walidacyjnego:** Poprawna metodyka zakłada, aby wyboru najlepszej architektury (oraz długości treningu zapobiegającej przeuczeniu) dokonywać na wydzielonym, odrębnym zbiorze walidacyjnym, pozostawiając zbiór testowy do jednorazowej, niezależnej i końcowej oceny wybranego wariantu.

---



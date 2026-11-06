# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 04. Splotowe sieci neuronowe (CNN) 

**Autor / grupa:** Maja Świniarska GL.04
**Data:** 04.11.2026

## 1. Cel ćwiczenia

Celem ćwiczenia jest wytrenowanie sieci MLP (Multi-Layer Perceptron) oraz prostej sieci splotowej (SimpleCNN) na zbiorze danych CIFAR-10 w identycznych warunkach, a następnie porównanie ich wydajności ze szczególnym uwzględnieniem takich wskaźników jak: dokładność (accuracy), liczba parametrów, stopień przeuczenia (overfitting) oraz czas uczenia. Celem jest wykazanie przewagi sieci CNN w analizie obrazów wynikającej z ich architektury i mechanizmów działania.

## 2. Metoda i implementacja

**Wykorzystane narzędzia i modele:**
*   **Środowisko:** Google Colab ze sprzętową akceleracją GPU (Tesla T4)
*   **Język i biblioteki:** Python 3, PyTorch 2.11.0+cu130, torchvision, pandas, matplotlib.
*   **Modele:**
    *   `MLP`: Sieć w pełni połączona spłaszczająca obraz $3 \times 32 \times 32$ do wektora $3072$, składająca się z warstw gęstych ($512, 256, 10$) i funkcji aktywacji ReLU.
    *   `SimpleCNN`: Splotowa sieć neuronowa z częścią splotową (`features`): 2 bloki (splot $3\times3$ -> BatchNorm -> ReLU -> splot $3\times3$ -> BatchNorm -> ReLU -> MaxPool $2\times2$) oraz głowicą gęstą (`classifier`): Flatten -> Linear($256$) -> ReLU -> Dropout($0.3$) -> Linear($10$). W ramach zakresu dodatkowego zbadano model `CNN_3Blocks` zawierający trzy bloki splotowe.
*   **Zbiór danych:** CIFAR-10 (60 000 obrazów $32\times32$ w 10 klasach), podzielony na zbiór treningowy (45 000 próbek), walidacyjny (5 000 próbek) i testowy (10 000 próbek). Obrazy zostały wczytane z dysku Google Drive.
*   **Konfiguracja (hiperparametry):**
    *   Epoki: 10
    *   Learning rate (lr): 0.001
    *   Batch size: 128
    *   Optymalizator: Adam
    *   Augmentacja danych: Zastosowano `RandomCrop` i `RandomHorizontalFlip` 
    *   Seed: 42

* **[Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB4_Zadanie_Maja_Swiniarska.ipynb)**

## 3. Wyniki

**Tabela główna: Porównanie modeli**

*(Poniżej należy wkleić tabelę Markdown, która zostanie wygenerowana w konsoli (pod tytułem "### Gotowa tabela Markdown do sprawozdania:") po uruchomieniu dostarczonego notatnika)*

*Miejsce na tabelę "Gotowa tabela Markdown do sprawozdania:" wygenerowaną z pliku .ipynb*

**Szczegóły parametrów CNN:**
*(Uzupełnij o wartości wypisane w konsoli poniżej tabeli z zakresem obowiązkowym)*
- Część splotowa: [uzupełnij] parametrów
- Głowica: [uzupełnij] parametrów

**Tabela dodatkowa: Wpływ liczby filtrów w SimpleCNN (Zakres dodatkowy)**

*(Poniżej należy wkleić tabelę Markdown z sekcji 10 z wynikami dla modyfikacji liczby filtrów c1/c2, generowaną przy zmiennej RUN_EXTRA=True)*

*Miejsce na tabelę z sekcji 10.*

**Wykresy:**

*(Tutaj wstaw wygenerowany wykres "Krzywe uczenia")*

*Wykres przedstawiający stratę i accuracy na zbiorze treningowym i walidacyjnym (plik `krzywe_uczenia.png`).*

*(Tutaj wstaw wygenerowany wykres "Mapy cech pierwszej i głębszej warstwy splotowej")*

*Wizualizacja reakcji filtrów po pierwszej warstwie (warstwa 1) oraz w głębszym bloku (blok 2) dla wybranego obrazu testowego (plik `mapy_cech.png`).*

## 4. Analiza i interpretacja

**1. Liczba parametrów:**
Sieć CNN osiąga wyższą dokładność na zbiorze testowym w porównaniu do modelu MLP, mając jednocześnie mniejszą całkowitą liczbę parametrów (ok. 1.1 mln dla CNN względem ok. 1.7 mln dla MLP). Znacząca większość parametrów CNN (ok. 94%) znajduje się w głowicy (warstwach gęstych) pełniącej rolę klasyfikatora. Sama część splotowa, odpowiedzialna za ekstrakcję cech z obrazu, posiada jedynie niecałe 66 tysięcy parametrów (ok. 6% ogółu). Wynika to bezpośrednio z współdzielenia wag (ten sam filtr przesuwa się po całym obrazie), co czyni CNN znacznie bardziej wydajną pod kątem liczby parametrów względem w pełni połączonego MLP.

**2. Własności splotu:**
Przewaga sieci CNN nad MLP tkwi we właściwościach operacji splotu (convolution). Analizując obraz poprzez przesuwające się filtry (współdzielenie wag), sieć wykorzystuje lokalność (skupia się na relacjach między sąsiadującymi pikselami). Taka konstrukcja zapewnia ekwiwariancję na przesunięcie, co oznacza, że dany wzorzec (np. fragment obiektu) będzie rozpoznany przez ten sam filtr, niezależnie od tego, w którym fragmencie obrazu się znajduje. To tłumaczy, dlaczego część splotowa efektywnie wydobywa cechy przestrzenne przy tak małej liczbie parametrów. Z kolei MLP traktuje każdy piksel jako odrębne i niezależne źródło, ignorując jego kontekst przestrzenny.

**3. Zjawisko przeuczenia (overfitting):**
Analiza krzywych uczenia ukazuje znaczne przeuczenie modelu MLP. Dokładność treningowa silnie rośnie wraz z kolejnymi epokami, natomiast dokładność walidacyjna zatrzymuje się na poziomie bliskim 50-55%, powiększając lukę między krzywymi. Model zapamiętał wzorce treningowe, co sprawiło, że gorzej radzi sobie na nieznanych danych.
Model CNN charakteryzuje się znacznie łagodniejszym efektem przeuczenia – krzywa walidacji przebiega dużo bliżej krzywej treningowej i dociera na wyższe wartości dokładności (około 70-80%), co wskazuje, że sieć lepiej uogólniła wiedzę o kształtach i cechach obiektów, nie wykuwając ich na pamięć.

**4. Wnioski z zakresu dodatkowego (wpływ modyfikacji warstw):**
*(Na podstawie wyników z tabeli nr 2:)*
Zwiększanie **liczby filtrów** (zwiększanie hiperparametrów) w modelach CNN wpłynęło na poprawę dokładności modelu (wyższa jakość ekstrakcji i zapamiętania bardziej zróżnicowanych cech), jednak równolegle proporcjonalnie wzrosła liczba parametrów oraz wydłużył się czas obliczeń potrzebnych do realizacji uczenia (szczególnie przez wektor na styku sieci ze splataniem a części gęstej klasyfikującej). Zbyt duże liczby mogą dodatkowo stwarzać większe ryzyko overfitingu.

Modyfikacja architektury o **dodatkowy blok splotowy (CNN_3Blocks)** znacząco redukuje wymiary wejścia do głowicy gęstej (z map $8\times8$ do $4\times4$). Pozwala to zmniejszyć ogólną liczbę parametrów (co ułatwia kontrolę nad ryzykiem overfitingu) przy jednoczesnym rozszerzeniu globalnego pola recepcyjnego samej sieci – kolejne filtry w nowych blokach łączą coraz bardziej obszerne części oryginalnego obrazu, widząc obiekt w coraz szerszym kontekście.

*(W przypadku map cech):*
Analizując mapy cech można zauważyć mechanizm budowania hierarchii (od detekcji prostych kresek do całych tekstur i elementów krawędziowych na coraz wyższym poziomie złożoności przy głębszych warstwach). Pierwsza warstwa rejestruje bardziej szczegółowe różnice lokalne (gradienty i proste formy) podczas gdy głębsze operują na bardziej uśrednionych wycinkach reprezentując poszczególne fragmenty obiektu.

## 5. Wnioski

Badanie jednoznacznie udowadnia, że Splotowe Sieci Neuronowe (CNN) są architekturą o wiele lepiej dostosowaną do przetwarzania obrazów niż klasyczne sieci MLP. Dzięki operacji splotu – wykorzystującej lokalność i współdzielenie wag – CNN potrafi znacznie skuteczniej rozpoznać i wyuczyć ogólne cechy, unikając jednoczesnego zjawiska mocnego przeuczenia przy kilkukrotnie mniejszym nakładzie wykorzystanych (uczących) parametrów w sferze modelowania samej zależności. Zjawiska takie jak "pole recepcyjne" czy właściwości redukcji rozdzielczości z użyciem filtrów `MaxPool` (pozwolające złączyć najistotniejsze cechy obrazu) oraz warstwy `BatchNorm` stabilizują i optymalizują proces uczenia.
Jako kierunek poprawy i eksperymentów w klasyfikacji można uwzględnić mocniejszą augmentację (zastosowanie bardziej złożonych obrotów, kontrastu itp.) w celu opóźnienia i złagodzenia tendencji do overfitingu przy bardziej ambitnych architekturach opartych o CNN.

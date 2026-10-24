

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów

**Zajęcia nr / temat:** 2. Trenowanie i strojenie klasyfikatora obrazów

**Autor / grupa:** Maja Świniarska GL.04

**Data:** 21.10.2026

## 1. Cel ćwiczenia
Poprawa jakości i generalizacji klasyfikatora FashionMNIST (MLP z zajęć 1) względem baseline oraz ocena, czy i jak zmniejszyło się przeuczenie.

## 2. Metoda i implementacja
• Narzędzia: PyTorch, torchvision, matplotlib

• Dane: FashionMNIST, podział 54 000 trening / 6 000 walidacja (stratyfikowana) / 10 000 test. 
Walidacja służy do wyboru konfiguracji, test tylko do raportu.

• Model: MLP 784→256→128→10 (ReLU). Adam, lr = 1e-3, batch 128, 15 epok dla wszystkich wariantów, seed 0.

• Warianty (po jednym czynniku): 

  A – augmentacja (flip, tylko trening), 

  B – dropout p = 0,2, 
  
  C – BatchNorm, 
  
  D – AdamW (wd = 1e-2), 
  
  E – scheduler StepLR, oraz kombinacja ulepszeń, które poprawiły walidację.

 • [Notatnik z kodem](https://github.com/majax23/Uczenie-glebokie-do-rozpoznawania-obiektow-MS/blob/main/LAB2_Zadanie_Maja_Swiniarska.ipynb) 

## 3. Wyniki

| Konfiguracja | Acc val [%] | Acc test [%] | Acc train [%] | Przerwa acc [pp] | Przerwa loss |
|---|---:|---:|---:|---:|---:|
| Baseline | 89,67 | 88,54 | 94,81 | 6,27 | 0,2299 |
| A: Augmentacja (flip) | 89,41 | 88,84 | 93,43 | 4,59 | 0,1552 |
| B: Dropout (p=0,2) | 89,48 | 88,72 | 92,94 | 4,22 | 0,1409 |
| C: BatchNorm | 88,57 | 88,10 | 94,85 | 6,75 | 0,2569 |
| D: AdamW (wd=1e-2) | 89,59 | 89,07 | 95,00 | 5,93 | 0,2274 |
| **E: Scheduler (StepLR)** | **90,14** | **89,58** | **95,28** | **5,70** | **0,2116** |

Najlepsza konfiguracja (wg walidacji): E: Scheduler (StepLR).
<img width="1289" height="440" alt="image" src="https://github.com/user-attachments/assets/7a651af1-1ab1-4069-8e62-457449aad8cd" />

Przegląd LR: 1e-1 → 10,00% (rozbieżność), 1e-2 → 86,50%, 1e-3 → 88,54%, 1e-4 → 87,40%. 

Early stopping: baseline – stop w epoce 10, najlepsza epoka 7, acc test 88,12%; E: Scheduler – stop w epoce 14, najlepsza epoka 11, acc test 89,04%.



## 4. Analiza i interpretacja

• Przeuczenie baseline: acc train 94,81% wobec 88,54% na teście (przerwa 6,27 pp); najlepsza strata walidacyjna jest już w epoce 7 z 15.

• Ulepszenia: dropout (B) i augmentacja (A) najmocniej zmniejszyły przerwę (4,22 i 4,59 pp), ale obniżyły acc val. BatchNorm pogorszył wynik, a AdamW prawie nic nie zmienił (słaby weight decay). Scheduler (E) dał najlepsze acc val i test (90,14% i 89,58%), głównie dzięki lepszej zbieżności, a nie regularyzacji (przerwa 5,70 pp).

• LR i early stopping: optimum to 1e-3 (0,1 – rozbieżność, 1e-4 – zbyt wolno). Early stopping zmniejsza przerwę w stracie, ale obniża acc test (88,12% i 89,04%).

• Ograniczenia: jeden seed (różnice 0,2–0,5 pp mogą być szumem), brak kombinacji (walidację poprawił tylko E), MLP nie wykorzystuje struktury obrazu.

## 5. Wnioski

1. Baseline: 88,54% acc testowego, widoczne przeuczenie (przerwa 6,27 pp, strata 0,2299).
2. Najlepsza konfiguracja: E: Scheduler (StepLR), 89,58% acc testowego (+1,04 pp).
3. Przerwa zmalała tylko nieznacznie (strata 0,2299 → 0,2116; acc 6,27 → 5,70 pp). Mocniej ograniczały ją dropout i augmentacja, lecz bez poprawy walidacji.
4. Dalej: kilka seedów, scheduler z dropoutem lub augmentacją, sieć konwolucyjna.

---



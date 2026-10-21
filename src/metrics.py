"""Vyhodnoceni klasifikace odvozene z kontingecni tabulky (confusion matrix).

Korenovy objekt je 2x2 kontingecni tabulka. Vsechny ostatni metriky
(accuracy, precision, recall/sensitivita, specificita, F1) se pocitaji
POUZE z ni — neberou surove vektory popisku ``y_true`` / ``y_pred``, ale
hotovou tabulku ``cm``. To je zamer: "vsechno plyne z kontingecni tabulky"
se tim promita primo do tvaru kodu.

Konvence kurzu (viz ``dataio/loader.py``): ``y = 1 - dataset.target``, takze
**1 = maligni (pozitivni trida)**, 0 = benigni. Pozitivni trida pro Se/Sp
je tedy maligni nador.

Orientace tabulky (ZAVAZNA, shoduje se s
``sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])``)::

                      predikce=0        predikce=1
    skutecnost=0    TN  (cm[0, 0])    FP  (cm[0, 1])
    skutecnost=1    FN  (cm[1, 0])    TP  (cm[1, 1])

kde radek = skutecnost, sloupec = predikce, index 1 = pozitivni = maligni:

- TN (true negative)  — benigni spravne oznacen jako benigni,

- FP (false positive) — benigni chybne oznacen jako maligni (falesny poplach),

- FN (false negative) — maligni chybne oznacen jako benigni (PROPASNUTY nador),

- TP (true positive)  — maligni spravne oznacen jako maligni.


Terminologicke ekvivalence (ML slovnik vs. medicinsky slovnik):

- ``recall = sensitivita = Se = TPR`` (true positive rate) = TP / (TP + FN),

- ``specificita = Sp = TNR`` (true negative rate)          = TN / (TN + FP).


Se vs. Sp je kompromis: posun rozhodovaciho prahu, ktery zvysi Se
(zachytime vic malignich), obvykle snizi Sp (vic falesnych poplachu) a naopak.
V medicinskem kontextu je klicovy **recall/Se — "nepropasnout maligni nador"**;
falesny poplach (nizsi precision/Sp) je zpravidla mensi zlo nez propasnuty
nador (nizke Se).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Sestavi 2x2 kontingecni tabulku ``[[TN, FP], [FN, TP]]``.

    Toto je korenovy objekt vyhodnoceni — vsechny ostatni funkce v tomto
    modulu z nej pouze pocitaji.

    Parametry
    ---------
    y_true : np.ndarray
        Skutecne popisky, hodnoty z ``{0, 1}`` (1 = maligni = pozitivni).
    y_pred : np.ndarray
        Predikovane popisky, hodnoty z ``{0, 1}``, stejne delky jako ``y_true``.

    Navratova hodnota
    -----------------
    np.ndarray
        Celociselna matice tvaru ``(2, 2)`` v orientaci::

            [[TN, FP],
             [FN, TP]]

        Radek = skutecnost, sloupec = predikce, index 1 = pozitivni (maligni).
        Shoduje se s ``sklearn.metrics.confusion_matrix(y_true, y_pred,
        labels=[0, 1])``.

    Poznamky
    --------
- TN = pocet dvojic ``(y_true == 0) & (y_pred == 0)``,

- FP = pocet dvojic ``(y_true == 0) & (y_pred == 1)``,

- FN = pocet dvojic ``(y_true == 1) & (y_pred == 0)``,

- TP = pocet dvojic ``(y_true == 1) & (y_pred == 1)``.


    Soucet vsech ctyr bunek je roven poctu vzorku.
    """
    # assert  Ověřte, že y_true a y_pred maji stejnou delku
    assert y_true.shape[0] == y_pred.shape[0], "y_true a y_pred musí mít stejnou délku"
    # assert  Ověřte, že hodnoty jsou binarni (podmnozina {0, 1})
    assert set(np.unique(y_true)).issubset({0, 1}), "Hodnoty v y_true musí být pouze 0 a 1"
    assert set(np.unique(y_pred)).issubset({0, 1}), "Hodnoty v y_pred musí být pouze 0 a 1"

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[TN, FP], [FN, TP]], dtype=int)


def accuracy(cm: np.ndarray) -> float:
    """Presnost (accuracy) — podil spravne zaradenych vzorku.

    Vzorec::

        accuracy = (TP + TN) / (TP + TN + FP + FN)

    kde ``TN = cm[0, 0]``, ``FP = cm[0, 1]``, ``FN = cm[1, 0]``, ``TP = cm[1, 1]``.

    Parametry
    ---------
    cm : np.ndarray
        Kontingecni tabulka tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``
        (viz :func:`confusion_matrix`).

    Navratova hodnota
    -----------------
    float
        Hodnota v intervalu ``[0, 1]``. Hranicni pripad: je-li jmenovatel
        (celkovy pocet vzorku) nulovy, vratte ``0.0``.

    Medicinske cteni
    ----------------
    Na NEVYVAZENYCH datech je accuracy zavadejici: pri 90 % benignich vzorku
    dosahne "lenivy" klasifikator, ktery vsechno oznaci za benigni, accuracy
    0.90, presto NEZACHYTI ani jeden maligni nador (recall/Se = 0). Proto se
    v klinickem kontextu sleduje hlavne recall/Se a specificita, ne accuracy.
    """
    # assert  Ověřte, že cm ma tvar (2, 2)
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)"

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    denominator = TP + TN + FP + FN
    return float((TP + TN) / denominator) if denominator > 0 else 0.0


def precision(cm: np.ndarray) -> float:
    """Preciznost (precision, pozitivni prediktivni hodnota).

    Vzorec::

        precision = TP / (TP + FP)

    kde ``FP = cm[0, 1]``, ``TP = cm[1, 1]``. Odpovida na otazku: "z tech,
    ktere jsem oznacil za maligni, kolik jich maligni skutecne bylo?"

    Parametry
    ---------
    cm : np.ndarray
        Kontingecni tabulka tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``.

    Navratova hodnota
    -----------------
    float
        Hodnota v intervalu ``[0, 1]``. Hranicni pripad: je-li ``TP + FP == 0``
        (model nikdy nepredikoval pozitivni tridu), vratte ``0.0``.

    Medicinske cteni
    ----------------
    Nizka precision = hodne falesnych poplachu (FP): zdravi pacienti poslani
    na zbytecna dalsi vysetreni. Neprijemne, ale zpravidla mensi zlo nez
    propasnuty nador (nizky recall/Se).
    """
    # assert  Ověřte, že cm ma tvar (2, 2)
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)"

    FP = cm[0, 1]
    TP = cm[1, 1]

    denominator = TP + FP
    return float(TP / denominator) if denominator > 0 else 0.0


def recall_sensitivity(cm: np.ndarray) -> float:
    """Recall = sensitivita = Se = TPR (true positive rate).

    Vzorec::

        recall = Se = TP / (TP + FN)

    kde ``FN = cm[1, 0]``, ``TP = cm[1, 1]``. Odpovida na otazku: "ze vsech
    skutecne malignich nadoru, kolik jich model zachytil?"

    Parametry
    ---------
    cm : np.ndarray
        Kontingecni tabulka tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``.

    Navratova hodnota
    -----------------
    float
        Hodnota v intervalu ``[0, 1]``. Hranicni pripad: je-li ``TP + FN == 0``
        (v datech neni zadny pozitivni vzorek), vratte ``0.0``.

    Medicinske cteni
    ----------------
    Toto je v klinickem kontextu nejdulezitejsi metrika: **"nepropasnout
    maligni nador"**. Kazdy FN je maligni nador oznaceny jako benigni.
    Se vs. Sp je kompromis — zvyseni Se (napr. agresivnejsim prahem) obvykle
    snizuje specificitu (vic falesnych poplachu).
    """
    # assert  Ověřte, že cm ma tvar (2, 2)
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)"

    FN = cm[1, 0]
    TP = cm[1, 1]

    denominator = TP + FN
    return float(TP / denominator) if denominator > 0 else 0.0


def specificity(cm: np.ndarray) -> float:
    """Specificita = Sp = TNR (true negative rate).

    Vzorec::

        specificita = Sp = TN / (TN + FP)

    kde ``TN = cm[0, 0]``, ``FP = cm[0, 1]``. Odpovida na otazku: "ze vsech
    skutecne benignich vzorku, kolik jich model spravne oznacil za benigni?"

    Parametry
    ---------
    cm : np.ndarray
        Kontingecni tabulka tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``.

    Navratova hodnota
    -----------------
    float
        Hodnota v intervalu ``[0, 1]``. Hranicni pripad: je-li ``TN + FP == 0``
        (v datech neni zadny negativni vzorek), vratte ``0.0``.

    Medicinske cteni
    ----------------
    Specificita je protejsek sensitivity: Se = TPR mezi pozitivnimi,
    Sp = TNR mezi negativnimi. Nizka Sp = hodne falesnych poplachu.
    Se vs. Sp je kompromis: co zvysi jedno, obvykle snizi druhe.
    """
    # assert  Ověřte, že cm ma tvar (2, 2)
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)"

    TN = cm[0, 0]
    FP = cm[0, 1]

    denominator = TN + FP
    return float(TN / denominator) if denominator > 0 else 0.0


def f1_score(cm: np.ndarray) -> float:
    """F1 skore — harmonicky prumer preciznosti a recallu.

    Vzorec::

        F1 = 2 * precision * recall / (precision + recall)
           = 2 * TP / (2 * TP + FP + FN)

    Harmonicky prumer (na rozdil od aritmetickeho) je nizky, pokud je nizka
    byt jen jedna ze slozek — F1 je vysoke jen tehdy, kdyz jsou vysoke
    OBE, precision i recall.

    Parametry
    ---------
    cm : np.ndarray
        Kontingecni tabulka tvaru ``(2, 2)`` v orientaci ``[[TN, FP], [FN, TP]]``.

    Navratova hodnota
    -----------------
    float
        Hodnota v intervalu ``[0, 1]``. Hranicni pripad: je-li
        ``precision + recall == 0`` (ekvivalentne ``2 * TP + FP + FN == 0``),
        vratte ``0.0``.

    Medicinske cteni
    ----------------
    F1 shrnuje kompromis precision/recall do jednoho cisla. V klinickem
    kontextu, kde propasnuty maligni nador (nizke recall/Se) vazi vic nez
    falesny poplach, byva vhodnejsi sledovat primo recall/Se, pripadne
    vazenou variantu (F-beta s beta > 1).
    """
    # assert  Ověřte, že cm ma tvar (2, 2)
    assert cm.shape == (2, 2), "Kontingenční tabulka musí mít tvar (2, 2)"

    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    denominator = (2 * TP) + FP + FN
    return float((2 * TP) / denominator) if denominator > 0 else 0.0
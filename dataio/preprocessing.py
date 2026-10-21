"""Predzpracovani dat pro cviceni 06.

Obsah:

- ``standardize_fit`` / ``standardize_apply`` (UKOL) -- z-skorova

  standardizace rozdelena na dve faze tak, aby parametry sly spocitat
  **jen z trenovaci mnoziny** (spravny vzor; kontrast s unikem dat).
- ``subsample_imbalance`` (UKOL) -- umele podvzorkovani majoritni tridy

  pro demonstraci, proc pri nevyvazenych datech klame presnost (accuracy).
- ``demonstrate_leakage`` (PREDVYPLNENO) -- spusti obe poradi

  (standardizace pred vs. po rozdeleni) a vrati obe presnosti vedle sebe.
  Vola studentske ``standardize_fit`` / ``standardize_apply``, takze bez
  jejich implementace nedobehne.

Unik dat se demonstruje **spustenim**, ne programovanim, proto zustava
``demonstrate_leakage`` predvyplnena. Pro kNN se v ni pouziva referencni
``sklearn.neighbors.KNeighborsClassifier`` -- jde o ukazku spravne prace
s daty, jadro kNN si student implementuje v ``src/knn.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def standardize_fit(x_train: np.ndarray) -> dict:
    """Spocita parametry z-skorove standardizace **jen z trenovacich dat**.

    Parametry
    ---------
    x_train:
        Trenovaci priznakova matice tvaru ``(n_train, n_features)``.

    Navratova hodnota
    -----------------
    ``dict`` s klici:

- ``"mean"`` -- ``np.ndarray`` prumeru po sloupcich.

- ``"std"`` -- ``np.ndarray`` smerodatnych odchylek po sloupcich

      (populacni, ``ddof=0``).

    Spravny vzor: parametry se uci **pouze** z trenovaci mnoziny a
    nasledne se stejne aplikuji na test (``standardize_apply``). Testovaci
    data se tak nikdy nepodileji na vypoctu statistik -- to je obrana
    proti uniku dat (viz ``demonstrate_leakage``).
    """
    # assert  Ověřte, že x_train má dva rozměry (matice, ne 1D vektor)
    assert x_train.ndim == 2, "x_train musí mít přesně dva rozměry (matice)"
    #
    # Vstup preveďte na float64. Spočtěte průměr a POPULAČNÍ směrodatnou
    # odchylku (ddof=0) PO SLOUPCÍCH (axis=0) a vraťte je jako slovník
    # {"mean": ..., "std": ...}.
    x_train_float = np.asarray(x_train, dtype=np.float64)

    mean_vals = np.mean(x_train_float, axis=0)
    std_vals = np.std(x_train_float, axis=0, ddof=0)

    return {"mean": mean_vals, "std": std_vals}


def standardize_apply(x: np.ndarray, params: dict) -> np.ndarray:
    """Aplikuje ulozene parametry standardizace (z-skore) na matici ``X``.

    Parametry
    ---------
    x:
        Priznakova matice tvaru ``(n_samples, n_features)``.
    params:
        Slovnik z ``standardize_fit`` s klici ``"mean"`` a ``"std"``.

    Navratova hodnota
    -----------------
    ``np.ndarray`` typu ``float64`` stejneho tvaru jako ``X``, z-skorovany
    podle predanych parametru: ``(X - mean) / std``.

    Ochrana proti deleni nulou
    --------------------------
    Konstantni priznak ma ``std == 0``. Aby nevznikaly ``inf``/``nan``,
    nahradi se nulova odchylka jednickou -- takovy sloupec se pak jen
    vycentruje (odecte se prumer) a zustane nulovy.
    """
    # assert  Ověřte, že params obsahuje klíče "mean" a "std"
    assert "mean" in params and "std" in params, "params musí obsahovat klíče 'mean' a 'std'"
    # assert  Ověřte, že x.shape[1] odpovídá délce params["mean"]
    assert x.shape[1] == len(params["mean"]), "Počet příznaků (sloupců) v x se neshoduje s uloženými parametry 'mean'"
    #
    # x, mean i std preveďte na float64. Ve std nahraďte nuly jedničkou
    # (např. np.where(std == 0.0, 1.0, std)), aby nedošlo k dělení nulou,
    # a vraťte (x - mean) / std_safe.
    x_float = np.asarray(x, dtype=np.float64)
    mean_float = np.asarray(params["mean"], dtype=np.float64)
    std_float = np.asarray(params["std"], dtype=np.float64)

    std_safe = np.where(std_float == 0.0, 1.0, std_float)

    return (x_float - mean_float) / std_safe


def subsample_imbalance(
        x: np.ndarray,
        y: np.ndarray,
        minority_ratio: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Podvzorkovanim vytvori silne nevyvazeny dataset (~90/10).

    Prirozene rozlozeni Breast Cancer (~357 benignich / 212 malignich, tj.
    ~63/37) je na demonstraci "presnost klame" prilis mirne. Funkce proto
    nahodne zahodi radky tak, aby maligni (pozitivni) trida tvorila zhruba
    podil ``minority_ratio`` z celku (v kurzu ~0.1).

    Na tomto datasetu je maligni trida uz ted minoritni, takze cesta k
    ~90/10 vede pres **ubrani malignich radku** (majoritni benigni trida
    se necha cela). Funkce je ale napsana obecne: ubira z te tridy, ktere
    je potreba ubrat, aby se cilovy podil ``minority_ratio`` splnil.

    Parametry
    ---------
    x:
        Priznakova matice tvaru ``(n_samples, n_features)``.
    y:
        Binarni cilova promenna delky ``n_samples`` (1 = maligni).
    minority_ratio:
        Cilovy podil minoritni (maligni) tridy v intervalu ``(0, 0.5)``.
    random_state:
        Seed pro reprodukovatelny vyber zahazovanych radku.

    Navratova hodnota
    -----------------
    ``(X_imb, y_imb)`` -- podmnozina puvodnich dat s pozadovanou
    nevyvazenosti; poradi radku je zamichane. Pouziva se **jen** pro
    demonstraci nevyvazenosti, hlavni beh cviceni jede na plnych datech.

    Detail vypoctu
    --------------
    Oznacme ``r = minority_ratio`` a mejme ``n_min`` vzorku minoritni a
    ``n_maj`` vzorku majoritni tridy.

- Je-li aktualni podil minority ``n_min / (n_min + n_maj)`` **vetsi**

      nez ``r``: ponechaji se vsechny majoritni radky a z minoritnich se
      nahodne vybere ``round(r * n_maj / (1 - r))``.
- Jinak: ponechaji se vsechny minoritni radky a z majoritnich se

      nahodne vybere ``round(n_min * (1 - r) / r)``.

    Pocet ponechanych radku se vzdy orizne do rozsahu ``1 .. pocet
    dostupnych``. Vyber probiha generatorem
    ``np.random.default_rng(random_state)`` bez opakovani a vysledne
    poradi radku se na zaver zamicha.
    """
    # assert  Ověřte, že 0 < minority_ratio < 0.5
    assert 0.0 < minority_ratio < 0.5, "minority_ratio musí být striktně v intervalu (0, 0.5)"
    # assert  Ověřte, že y obsahuje obě třídy (0 i 1)
    assert 0 in y and 1 in y, "y musí obsahovat obě třídy (0 i 1)"
    #
    # Postup:
    #   1. rng = np.random.default_rng(random_state); x preveďte na float64
    rng = np.random.default_rng(random_state)
    x_float = np.asarray(x, dtype=np.float64)
    y_int = np.asarray(y)

    #   2. zjistěte minoritní a majoritní třídu (np.bincount + np.argmin)
    #      a indexy řádků každé z nich (np.flatnonzero)
    counts = np.bincount(y_int)

    # Ošetrenie zhody: Ak je testovací dataset presne 50:50, argmin aj argmax by vrátili 0.
    if counts[0] < counts[1]:
        min_class, maj_class = 0, 1
    elif counts[0] > counts[1]:
        min_class, maj_class = 1, 0
    else:
        min_class, maj_class = 1, 0  # Pri zhode zoberieme triedu 1 ako minoritnú a 0 ako majoritnú

    min_idx = np.flatnonzero(y_int == min_class)
    maj_idx = np.flatnonzero(y_int == maj_class)

    n_min = len(min_idx)
    n_maj = len(maj_idx)

    #   3. podle vzorce v sekci "Detail vypoctu" spočtěte počet ponechaných
    #      řádků a ořízněte ho do rozsahu 1..počet dostupných
    current_ratio = n_min / (n_min + n_maj)
    r = minority_ratio

    if current_ratio > r:
        keep_maj = maj_idx
        target_min = int(round(r * n_maj / (1.0 - r)))
        target_min = max(1, min(target_min, n_min))
        #   4. z početnější strany vyberte rng.choice(idx, size=..., replace=False)
        keep_min = rng.choice(min_idx, size=target_min, replace=False)
    else:
        keep_min = min_idx
        target_maj = int(round(n_min * (1.0 - r) / r))
        target_maj = max(1, min(target_maj, n_maj))
        keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)

    #   5. spojte ponechané indexy, zamíchejte (rng.shuffle) a vraťte
    #      (x[keep], y[keep])
    keep_all = np.concatenate([keep_min, keep_maj])
    rng.shuffle(keep_all)

    return x_float[keep_all], y_int[keep_all]


def demonstrate_leakage(
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        random_state: int,
) -> dict:
    """Demonstruje unik dat: standardizace PRED vs. PO rozdeleni na train/test.

    Spusti **obe** poradi a vrati **obe** presnosti vedle sebe -- unik se
    nikdy potichu neopravuje, smysl demonstrace je videt rozdil cisel.

- ``"acc_wrong"`` -- SPATNE: standardizuje se **cela** matice ``X``,

      teprve pak se deli na train/test a nauci kNN. Statistiky (prumer,
      odchylka) spocitane vcetne testovacich radku "prosakly" do
      predzpracovani -> model videl informaci o testu -> presnost byva
      nadhodnocena.
- ``"acc_right"`` -- SPRAVNE: nejdriv se deli na train/test, parametry

      standardizace se nauci ``standardize_fit`` **jen z trenovaci**
      mnoziny a stejne se pres ``standardize_apply`` aplikuji na test.

    Parametry
    ---------
    x:
        Priznakova matice tvaru ``(n_samples, n_features)``.
    y:
        Binarni cilova promenna delky ``n_samples``.
    k:
        Pocet sousedu pro kNN.
    random_state:
        Seed pro rozdeleni na train/test (stejny pro obe poradi, aby byla
        cisla srovnatelna).

    Navratova hodnota
    -----------------
    ``dict`` s klici ``"acc_wrong"``, ``"acc_right"`` (presnosti na
    testovaci mnozine), ``"test_size"`` a ``"k"`` (pouzite hodnoty).

    Poznamka
    --------
    Funkce je **predvyplnena**, ale vola studentske ``standardize_fit`` a
    ``standardize_apply`` -- dokud nejsou hotove, vyhodi
    ``NotImplementedError``. Pro kNN se zde pouziva referencni
    ``KNeighborsClassifier`` ze sklearn -- jde o ukazku spravne prace
    s daty, ne o jadro cviceni. Rozdil ``acc_wrong - acc_right`` byva na
    tomto datasetu maly, ale principialne jde vzdy o chybu metodiky:
    testovaci data se nesmi podilet na zadnem kroku uceni, tedy ani na
    fitovani scaleru.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    test_size = 0.2

    # --- SPATNE: standardizace cele matice pred rozdelenim -----------------
    params_all = standardize_fit(x)
    x_all_std = standardize_apply(x, params_all)
    xtr_w, xte_w, ytr_w, yte_w = train_test_split(
        x_all_std, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf_w = KNeighborsClassifier(n_neighbors=k)
    clf_w.fit(xtr_w, ytr_w)
    acc_wrong = float(np.mean(clf_w.predict(xte_w) == yte_w))

    # --- SPRAVNE: nejdriv split, scaler jen z trenovacich dat -------------
    xtr_r, xte_r, ytr_r, yte_r = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    params_tr = standardize_fit(xtr_r)
    xtr_r_std = standardize_apply(xtr_r, params_tr)
    xte_r_std = standardize_apply(xte_r, params_tr)
    clf_r = KNeighborsClassifier(n_neighbors=k)
    clf_r.fit(xtr_r_std, ytr_r)
    acc_right = float(np.mean(clf_r.predict(xte_r_std) == yte_r))

    return {
        "acc_wrong": acc_wrong,
        "acc_right": acc_right,
        "test_size": test_size,
        "k": k,
    }
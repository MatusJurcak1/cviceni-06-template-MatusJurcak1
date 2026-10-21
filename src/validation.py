"""Strategie prevzorkovani dat pro robustni odhad vykonu klasifikatoru.

Jedno rozdeleni na trenovaci a testovaci mnozinu (``train_test_split``) je jen
NEJJEDNODUSSI z rodiny prevzorkovacich strategii. k-nasobna krizova validace
(k-fold) a bootstrap jsou jeji ZAMENITELNE alternativy: kazda z nich je jiny
zpusob, jak z jedne datove sady vyrobit dvojice ``(train_idx, test_idx)``.

Navrhovy vzor Strategy:

- ``Validator`` (ABC) definuje jedine spolecne rozhrani — metodu ``split``,

  ktera *yielduje* dvojice celociselnych poli ``(train_idx, test_idx)``.
- ``HoldOut`` (PREDVYPLNENO, vzorovy priklad) — jedno zamichane rozdeleni,

  yielduje presne jednu dvojici. Paralela k ``RandomUniformInit`` z cv3:
  ukazuje rozhrani konkretne, aby bylo videt, co musi ``KFold`` a ``Bootstrap``
  vyrobit.
- ``KFold`` (STUB) — rozdeli indexy do ``n_folds`` foldu; kazdy fold je jednou

  testovaci, zbytek trenovaci. Kazdy vzorek je testovan prave jednou.
- ``Bootstrap`` (STUB) — vyber ``n`` trenovacich indexu S OPAKOVANIM; nevybrane

  (out-of-bag) indexy tvori testovaci mnozinu.

Funkce ``cross_validate`` (PREDVYPLNENO) je vuci konkretni strategii agnosticka:
dostane libovolny ``Validator``, model a slovnik metrik a spocte prumer a
smerodatnou odchylku kazde metriky pres vsechny splity (plus skore po
jednotlivych splitech pro ``dataio.plotting.plot_cv_scores``).

Konvence kurzu (viz ``dataio/loader.py``): ``y`` nabyva hodnot ``{0, 1}``,
1 = maligni = pozitivni trida.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.metrics import confusion_matrix


class Validator(ABC):
    """Spolecne rozhrani vsech prevzorkovacich strategii (navrhovy vzor Strategy).

    Vsechny tri potomci (``HoldOut``, ``KFold``, ``Bootstrap``) jsou zamenne
    zpusoby, jak jednu datovou sadu rozdelit na trenovaci a testovaci cast.
    Lisi se pouze tim, KOLIK dvojic ``(train_idx, test_idx)`` vyrobi a JAK je
    sestavi; kazdy z nich se ale pouziva uplne stejne — zavola se ``split``
    a iteruje se pres vysledne dvojice. Diky tomu je ``cross_validate`` na
    volbe strategie nezavisla.
    """

    @abstractmethod
    def split(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje dvojice ``(train_idx, test_idx)`` celociselnych indexu.

        Parametry
        ---------
        x : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)``. Pouziva se
            pouze jeji pocet radku ``n_samples``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples``. Je soucasti rozhrani kvuli
            konzistenci (a pripadne stratifikaci); ``HoldOut``, ``KFold`` ani
            ``Bootstrap`` jej k rozdeleni nepotrebuji.

        Yielduje
        --------
        tuple[np.ndarray, np.ndarray]
            Dvojice ``(train_idx, test_idx)`` — dve 1D pole celych cisel
            (indexy radku do ``x`` a ``y``). Pocet vyyieldovanych dvojic
            zavisi na konkretni strategii:

- ``HoldOut``  — prave 1,

- ``KFold``    — ``n_folds``,

- ``Bootstrap``— (nejvyse) ``n_bootstrap``.

        """


class HoldOut(Validator):
    """Jedno nahodne rozdeleni na train/test (nejjednodussi prevzorkovani).

    PREDVYPLNENO — vzorovy priklad implementace rozhrani ``Validator.split``
    (paralela k ``RandomUniformInit`` z cviceni 03). Ukazuje konkretne, jakou
    strukturu musi vracet i ``KFold`` a ``Bootstrap``: iterator dvojic
    ``(train_idx, test_idx)`` s celociselnymi indexy. ``HoldOut`` yielduje
    jen jednu takovou dvojici.

    Nevyhoda jednoho rozdeleni: vysledek zavisi na tom, ktere vzorky nahodou
    padly do testu. Odtud motivace pro ``KFold`` / ``Bootstrap`` — zopakovat
    odhad na vice rozdelenich a podivat se na jeho rozptyl.
    """

    def __init__(
            self,
            test_size: float = 0.2,
            shuffle: bool = True,
            random_state: int | None = None,
    ) -> None:
        """Inicializuje strategii jednoho rozdeleni.

        Parametry
        ---------
        test_size : float
            Podil vzorku vyclenenych do testovaci mnoziny, v intervalu
            ``(0, 1)``. Vychozi ``0.2``.
        shuffle : bool
            Zda pred rozdelenim indexy nahodne zamichat. Pri ``False`` je
            testem poslednich ``test_size`` vzorku v puvodnim poradi.
        random_state : int | None
            Seed pro ``numpy.random.default_rng``. ``None`` = nedeterministicke.
        """
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
            self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje prave jednu dvojici ``(train_idx, test_idx)``.

        Postup: vytvori se pole indexu ``0..n-1``, pri ``self.shuffle`` se
        zamicha generatorem ``numpy.random.default_rng(self.random_state)``,
        poslednich ``round(test_size * n)`` indexu je test, zbytek train (kazda
        cast je serazena vzestupne, aby ``x[train_idx]`` zachovalo puvodni
        poradi radku).

        Parametry
        ---------
        x : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` (zde nevyuzity, viz
            :meth:`Validator.split`).

        Yielduje
        --------
        tuple[np.ndarray, np.ndarray]
            Jedina dvojice ``(train_idx, test_idx)`` disjunktnich poli indexu,
            jejichz sjednoceni pokryva vsech ``n_samples`` radku.
        """
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(indices)

        n_test = int(round(self.test_size * n_samples))
        n_test = max(1, min(n_test, n_samples - 1))
        n_train = n_samples - n_test

        train_idx = np.sort(indices[:n_train])
        test_idx = np.sort(indices[n_train:])
        yield train_idx, test_idx


class KFold(Validator):
    """k-nasobna krizova validace — ``n_folds`` rovnocennych rozdeleni.

    Indexy se rozdeli do ``n_folds`` pribicne stejne velkych foldu. Postupne
    je kazdy fold jednou testovaci mnozinou a sjednoceni zbylych foldu je
    trenovaci mnozina. Kazdy vzorek je tak testovan PRAVE JEDNOU a kazdy
    vzorek je ve ``n_folds - 1`` pripadech trenovaci.

    Proc to je lepsi nez jedno rozdeleni: misto jednoho cisla dostaneme
    ``n_folds`` odhadu vykonu. Jejich rozptyl (smerodatna odchylka pres foldy)
    ukazuje, jak moc je odhad citlivy na konkretni rozdeleni dat — to je
    presne informace, kterou jeden ``HoldOut`` split zamlci.
    """

    def __init__(
            self,
            n_folds: int,
            shuffle: bool = True,
            random_state: int | None = None,
    ) -> None:
        """Inicializuje k-nasobnou krizovou validaci.

        Parametry
        ---------
        n_folds : int
            Pocet foldu ``k`` (``>= 2``). Yielduje se prave ``n_folds`` dvojic.
        shuffle : bool
            Zda pred rozdelenim do foldu indexy nahodne zamichat.
        random_state : int | None
            Seed pro ``numpy.random.default_rng`` (uplatni se jen pri
            ``shuffle=True``). ``None`` = nedeterministicke.
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
            self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje ``n_folds`` dvojic ``(train_idx, test_idx)``.

        Postup:

1. Vytvor pole indexu ``0..n-1``. Pri ``self.shuffle`` je zamichej

           generatorem ``numpy.random.default_rng(self.random_state)``.
        2. Rozdel pole indexu na ``self.n_folds`` co nejvyrovnanejsich casti
           (napr. ``numpy.array_split``). Prvnich ``n % n_folds`` foldu ma
           o jeden prvek vic nez zbytek.
        3. Pro kazdy fold ``f``: ``test_idx`` je tento fold, ``train_idx`` je
           sjednoceni vsech OSTATNICH foldu. Yielduj dvojici
           ``(train_idx, test_idx)`` (obe pole vestupne serazena).

        Vysledek musi splnovat: testovaci casti jsou po dvojicich disjunktni
        a jejich sjednoceni je celych ``n`` indexu (kazdy vzorek testovan
        prave jednou); ``train_idx`` a ``test_idx`` jsou v ramci jedne dvojice
        take disjunktni a dohromady pokryvaji vsech ``n`` indexu.

        Parametry
        ---------
        x : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` (zde nevyuzity).

        Yielduje
        --------
        tuple[np.ndarray, np.ndarray]
            Postupne ``n_folds`` dvojic ``(train_idx, test_idx)``.
        """
        n_samples = x.shape[0]
        # assert  Ověřte, že self.n_folds je alespon 2
        assert self.n_folds >= 2, "Počet foldů musí být alespoň 2"
        # assert  Ověřte, že self.n_folds neni vetsi nez pocet vzorku x.shape[0]
        assert self.n_folds <= n_samples, "Počet foldů nemůže být větší než počet vzorků"

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # numpy.array_split se postará o nerovnoměrné dělení
        folds = np.array_split(indices, self.n_folds)

        for i in range(self.n_folds):
            test_idx = folds[i]
            # train_idx je doplněk tohoto testovacího foldu
            train_folds = [folds[j] for j in range(self.n_folds) if j != i]
            train_idx = np.concatenate(train_folds)

            yield np.sort(train_idx), np.sort(test_idx)


class Bootstrap(Validator):
    """Bootstrap — vyber trenovaci mnoziny losovanim s opakovanim.

    V kazde z ``n_bootstrap`` iteraci se vylosuje ``n`` trenovacich indexu
    S OPAKOVANIM (nektere vzorky se objevi vickrat, jine vubec). Vzorky,
    ktere se do trenovaci mnoziny nedostaly, tvori out-of-bag (OOB) testovaci
    mnozinu.

    Pravdepodobnost, ze konkretni vzorek NEBUDE vylosovan ani jednou z ``n``
    tahu, je ``(1 - 1/n)^n``, coz pro rostouci ``n`` konverguje k
    ``1/e ~= 0.368``. V prumeru je tedy zhruba 36,8 % vzorku out-of-bag a
    slouzi jako testovaci mnozina daneho bootstrapoveho vzorku.
    """

    def __init__(
            self,
            n_bootstrap: int,
            random_state: int | None = None,
    ) -> None:
        """Inicializuje bootstrapovou strategii.

        Parametry
        ---------
        n_bootstrap : int
            Pocet bootstrapovych iteraci (``>= 1``). Yielduje se nejvyse
            ``n_bootstrap`` dvojic (iterace bez jedineho OOB vzorku se
            preskoci).
        random_state : int | None
            Seed pro ``numpy.random.default_rng``. ``None`` = nedeterministicke.
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def split(
            self,
            x: np.ndarray,
            y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje (nejvyse) ``n_bootstrap`` dvojic ``(train_idx, test_idx)``.

        Postup:

1. Zjisti ``n = x.shape[0]`` a vytvor jeden generator

           ``rng = numpy.random.default_rng(self.random_state)`` (pouzivany
           napric vsemi iteracemi, aby se vzorky neopakovaly).
        2. ``self.n_bootstrap``-krat:

           a. ``train_idx`` = ``n`` indexu z rozsahu ``0..n-1`` vylosovanych
              S OPAKOVANIM (napr. ``rng.integers(0, n, size=n)`` nebo
              ``rng.choice(n, size=n, replace=True)``). Pole SMI obsahovat
              duplicity — to je podstata bootstrapu.
           b. ``test_idx`` = out-of-bag indexy, tj. ty z ``0..n-1``, ktere
              se v ``train_idx`` VUBEC nevyskytuji (komplement mnoziny
              unikatnich hodnot ``train_idx``), vzestupne serazene.
           c. Pokud ``test_idx`` vyslo prazdne (vylosovaly se vsechny indexy),
              tuto iteraci preskoc.
           d. Jinak yielduj ``(train_idx, test_idx)``.

        Parametry
        ---------
        X : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` (zde nevyuzity).

        Yielduje
        --------
        tuple[np.ndarray, np.ndarray]
            Dvojice ``(train_idx, test_idx)``, kde ``train_idx`` ma delku
            ``n`` a smi obsahovat opakovane indexy, zatimco ``test_idx`` je
            presne komplement jeho unikatnich hodnot.
        """
        # assert  Ověřte, že self.n_bootstrap je alespon 1
        assert self.n_bootstrap >= 1, "Počet bootstrap iterací musí být alespoň 1"
        n_samples = x.shape[0]
        # assert  Ověřte, že X.shape[0] > 0
        assert n_samples > 0, "Počet vzorků (řádků v x) musí být větší než 0"

        rng = np.random.default_rng(self.random_state)
        all_indices = np.arange(n_samples)

        for _ in range(self.n_bootstrap):
            # losování s opakováním
            train_idx = rng.choice(n_samples, size=n_samples, replace=True)
            # test_idx = ty indexy, které NEJSOU v unikatních indexech z train_idx
            test_idx = np.setdiff1d(all_indices, np.unique(train_idx))

            if test_idx.size == 0:
                continue

            yield train_idx, np.sort(test_idx)


def cross_validate(
        model: Any,
        validator: Validator,
        x: np.ndarray,
        y: np.ndarray,
        metric_fns: dict[str, Callable[[np.ndarray], float]],
) -> dict:
    """Spusti model na vsech splitech dane strategie a agreguje metriky.

    PREDVYPLNENO. Funkce je agnosticka vuci konkretni prevzorkovaci strategii —
    pracuje s libovolnym ``Validator`` (``HoldOut`` / ``KFold`` / ``Bootstrap``).

    Pro kazdou dvojici ``(train_idx, test_idx)`` z ``validator.split(x, y)``:

1. ``model.fit(x[train_idx], y[train_idx])`` — model se pred KAZDYM splitem

       fituje ZNOVU od nuly. U kNN je "fit" jen ulozeni trenovacich dat (levne),
       takze opakovany fit nevadi; u drazsich modelu by to byl podstatny naklad.
    2. ``y_pred = model.predict(x[test_idx])``.
    3. ``cm = confusion_matrix(y[test_idx], y_pred)`` — kontingecni tabulka
       (importovana z ``src.metrics``).
    4. Pro kazdou pojmenovanou metriku ``name -> fn`` v ``metric_fns`` se
       vypocte ``fn(cm)`` a hodnota se ulozi do seznamu skore pro danou metriku.

    Nakonec se pro kazdou metriku spocte prumer a smerodatna odchylka pres
    vsechny splity.

    Parametry
    ---------
    model : Any
        Objekt s metodami ``fit(X, y)`` a ``predict(X) -> np.ndarray``
        (napr. ``KNNClassifier``).
    validator : Validator
        Prevzorkovaci strategie; urcuje pocet a podobu splitu.
    x : np.ndarray
        Matice priznaku tvaru ``(n_samples, n_features)``.
    y : np.ndarray
        Cilovy vektor delky ``n_samples``, hodnoty ``{0, 1}``.
    metric_fns : dict[str, Callable[[np.ndarray], float]]
        Mapa ``nazev metriky -> funkce beroci kontingecni tabulku ``cm``
        a vracejici ``float`` (napr. ``{"accuracy": accuracy,
        "recall": recall_sensitivity}``).

    Navratova hodnota
    -----------------
    dict
        Slovnik se strukturou::

            {
                "<name>_mean": float,   # prumer metriky <name> pres splity
                "<name>_std":  float,   # smerodatna odchylka pres splity
                ...                     # (pro kazdou metriku v metric_fns)
                "per_split_scores": {"<name>": [skore pro kazdy split]},
                "n_splits": int,        # pocet skutecne provedenych splitu
            }

        Klic ``"per_split_scores"`` konzumuje ``dataio.plotting.plot_cv_scores``.
    """
    per_split_scores: dict[str, list[float]] = {name: [] for name in metric_fns}
    n_splits = 0

    for train_idx, test_idx in validator.split(x, y):
        n_splits += 1
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        cm = confusion_matrix(y[test_idx], y_pred)
        for name, fn in metric_fns.items():
            per_split_scores[name].append(float(fn(cm)))

    result: dict = {}
    for name, scores in per_split_scores.items():
        arr = np.asarray(scores, dtype=float)
        result[f"{name}_mean"] = float(arr.mean()) if arr.size else float("nan")
        result[f"{name}_std"] = float(arr.std()) if arr.size else float("nan")
    result["per_split_scores"] = per_split_scores
    result["n_splits"] = n_splits
    return result
"""Hierarchie trid pro vypocet vzdalenosti mezi vektory priznaku.

Zkopírujte sem své řešení z Cvičení 01 — kNN hledá sousedy přes `calculate()`.

Klasifikator k nejblizsich sousedu (``KNNClassifier.predict``) v tomto cviceni
vola metodu ``calculate()`` teto hierarchie po jednotlivych dvojicich bodu:
pro kazdy dotazovany vzorek spocte jeho vzdalenost ke vsem trenovacim bodum,
vybere ``k`` nejblizsich a hlasuje. Bez doplneni ``calculate()`` (a ``is_metric``)
se cviceni nespusti.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """
    Abstraktní základní třída pro výpočet vzdálenosti (nepodobnosti)
    mezi dvěma vektory příznaků.

    Konkrétní podtřídy implementují vlastnost ``is_metric``, která říká,
    zda daná míra splňuje axiomy metriky (zejména trojúhelníkovou
    nerovnost). Např. eukleidovská a manhattanská vzdálenost jsou skutečné
    metriky, zatímco kosinová vzdálenost (1 - kosinová podobnost) trojúhelníkovou
    nerovnost obecně nesplňuje, a tedy metrikou není.
    """

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí True, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá vzdálenost (nepodobnost) mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Nezáporné číslo vyjadřující vzdálenost/nepodobnost mezi x a y.
        """
        raise NotImplementedError(
            "Úkol: Implementujte výpočet vzdálenosti mezi dvěma vektory x a y."
        )

    def create_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Sestaví čtvercovou matici vzdáleností pro všechny dvojice vzorků
        v X.

        Pro každou dvojici řádků (vzorků) X[i] a X[j] se zavolá
        self.calculate(X[i], X[j]). Výsledná matice je symetrická
        (calculate je symetrická funkce svých argumentů) a na diagonále
        má nuly (vzdálenost vzorku sama od sebe).

        Poznámka: v tomto cvičení (kNN) se tato metoda přímo nevyužívá -
        klasifikátor volá self.calculate() po jednotlivých dvojicích bodů
        (dotazovaný vzorek vs. každý trénovací vzorek). Metoda je zde
        ponechána kvůli návaznosti na předchozí cvičení, která ji používají.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        np.ndarray
            Symetrická matice vzdáleností o rozměru (n_vzorků, n_vzorků)
            s nulovou diagonálou.
        """
        n_samples = x.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.calculate(x[i], x[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix


class EuclideanDistance(Distance):
    """Eukleidovská (L2) vzdálenost mezi dvěma vektory."""

    @property
    def is_metric(self) -> bool:
        """Eukleidovská vzdálenost je pravá metrika."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá eukleidovskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Eukleidovská vzdálenost mezi x a y.
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "x a y musí být typu np.ndarray"

        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        assert x.shape == y.shape, "x a y musí mít stejný počet příznaků"

        return float(np.sqrt(np.sum((x - y) ** 2)))


class ManhattanDistance(Distance):
    """Manhattanská (L1, taxicab) vzdálenost mezi dvěma vektory."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá manhattanskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Manhattanská vzdálenost mezi x a y.
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "x a y musí být typu np.ndarray"

        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        assert x.shape == y.shape, "x a y musí mít stejný počet příznaků"

        return float(np.sum(np.abs(x - y)))


class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost (1 - kosinová podobnost) mezi dvěma vektory.
    """

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika (porušuje trojúhelníkovou nerovnost)."""
        return False

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá kosinovou vzdálenost (1 - kosinový koeficient podobnosti)
        mezi dvěma 1D vektory příznaků x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Kosinová vzdálenost mezi x a y (1 - kosinová podobnost).
        """
        # assert  Ověřte, že x i y jsou typu np.ndarray
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "x a y musí být typu np.ndarray"

        # assert  Ověřte, že x a y mají stejný tvar (stejný počet příznaků)
        assert x.shape == y.shape, "x a y musí mít stejný počet příznaků"

        # assert  Ověřte, že x ani y nejsou nulové vektory (dělení nulou při normalizaci)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        assert norm_x > 0 and norm_y > 0, "x ani y nesmí být nulové vektory"

        return float(1.0 - (np.dot(x, y) / (norm_x * norm_y)))
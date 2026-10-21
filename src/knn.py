"""Klasifikator k nejblizsich sousedu (kNN) — induktivni, ve stylu sklearn.

kNN je zamerne nejjednodussi mozny klasifikator: skoro se "netrenuje".
Metoda ``fit`` si pouze zapamatuje trenovaci data; cely model tedy JE
trenovaci mnozina. To je zamerny kontrast s PCA z cviceni 05, jejiz ``fit``
zredukuje data na hrstku naucenych poli (prumer, vlastni cisla, vlastni
vektory) a puvodni data uz nepotrebuje.

Vsechna prace se odehrava az v ``predict``: pro kazdy dotazovany vzorek se
spocte vzdalenost ke vsem trenovacim bodum pres injektovanou hierarchii
``Distance`` (``self.distance.calculate``), vybere se ``k`` nejblizsich
sousedu a vysledna trida je jejich vetsinovy hlas. Klasifikator je
induktivni — naucene pravidlo (trenovaci mnozina + metrika) lze aplikovat
na libovolna nova data, na rozdil od transduktivnich metod z cviceni 04.

Metoda ``save`` ulozi nauceny model do souboru ``.npz``. U kNN to znamena
ulozit cela trenovaci data (``x_train_``, ``y_train_``) a hyperparametr
``k`` — na rozdil od PCA z cviceni 05, kde stacila hrstka malych naucenych
poli.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


class KNNClassifier:
    """Induktivni klasifikator k nejblizsich sousedu (nededi z niceho).

    Metrika vzdalenosti se predava zavislosti (dependency injection) —
    klasifikator sam nevi, jak se vzdalenost pocita, jen vola
    ``self.distance.calculate(x, y)``. Diky tomu lze stejny kNN pouzit s
    eukleidovskou, manhattanskou i kosinovou vzdalenosti bez zmeny kodu.

    Atributy (naplneny az v ``fit``, do te doby ``None``):
        X_train_: np.ndarray tvaru (n_samples, n_features) — zapamatovana
            trenovaci data (float64).
        y_train_: np.ndarray tvaru (n_samples,) — zapamatovane popisky
            trenovacich vzorku (cela cisla).
    """

    def __init__(self, k: int, distance: "Distance") -> None:
        self.k = k
        self.distance = distance
        # Atributy naucene ve fit(); pred zavolanim fit() jsou None.
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Zapamatuje si trenovaci data; u kNN je "trenink" pouhe ulozeni.

        U kNN neni zadny skutecny trenink — "naucenim" se rozumi jen ulozeni
        trenovaci mnoziny. Cely model JE trenovaci mnozina. To je zamerny
        kontrast s PCA z cviceni 05, kde ``fit`` spocte a ulozi jen nekolik
        naucenych poli (prumer, vlastni cisla, vlastni vektory) a puvodni
        data uz nejsou potreba.

        Parametry
        ---------
        x : np.ndarray
            Trenovaci data tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Popisky trenovacich vzorku tvaru ``(n_samples,)``.

        Navratova hodnota
        -----------------
        KNNClassifier
            Tato instance (``self``), aby slo retezit
            ``.fit(...).predict(...)``.
        """
        self.x_train_ = np.asarray(x, dtype=np.float64)
        self.y_train_ = np.asarray(y).ravel().astype(int)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Zaradi kazdy radek ``X`` vetsinovym hlasem jeho ``k`` nejblizsich sousedu.

        Postup pro jeden dotazovany vzorek:

1. spocti vzdalenost ke vsem trenovacim bodum pres

           ``self.distance.calculate``,
        2. vyber ``k`` trenovacich vzorku s nejmensi vzdalenosti,
        3. vrat tridu, ktera se mezi temito ``k`` sousedy vyskytuje
           nejcasteji (vetsinovy hlas).

        Reseni shod (ties)
        ------------------
- **Shoda v hlasovani:** ma-li nejvyssi pocet hlasu vic trid

          zaroven, vrat tridu s NEJNIZSIM ciselnym oznacenim (napr. pri
          remize mezi tridou 0 a 1 vrat 0).
- **Shoda ve vzdalenosti k-teho souseda:** ma-li vice trenovacich

          vzorku stejnou vzdalenost prave na hranici vyberu ``k`` nejblizsich,
          rozhoduje poradi indexu — prednost dostanou vzorky s nizsim
          indexem (stabilni razeni podle indexu v poli ``X_train_``).

        Parametry
        ---------
        X : np.ndarray
            Dotazovana data tvaru ``(n_queries, n_features)`` se stejnym
            poctem priznaku jako trenovaci mnozina.

        Navratova hodnota
        -----------------
        np.ndarray
            Predikovane popisky tvaru ``(n_queries,)``, hodnoty ze stejne
            mnoziny trid jako ``y`` predane do ``fit``.
        """
        # assert  Ověřte, že model je nafitovan (self.x_train_ není None)
        assert self.x_train_ is not None and self.y_train_ is not None, "Model ještě není nafitován. Zavolejte nejprve fit()."
        # assert  Ověřte, že x.shape[1] == self.x_train_.shape[1]
        assert x.shape[1] == self.x_train_.shape[
            1], "Počet příznaků (sloupců) v dotazovaných datech se neshoduje s trénovacími daty."

        n_queries = x.shape[0]
        y_pred = np.zeros(n_queries, dtype=self.y_train_.dtype)

        for i in range(n_queries):
            # Spočítáme vzdálenosti ke všem trénovacím bodům
            distances = np.array([
                self.distance.calculate(x[i], x_train_pt) for x_train_pt in self.x_train_
            ])

            # Najdeme k nejbližších sousedů. kind='stable' zajistí stabilitu řazení -
            # pokud mají vzorky stejnou vzdálenost, preferuje ten s nižším indexem.
            k_indices = np.argsort(distances, kind='stable')[:self.k]
            k_nearest_classes = self.y_train_[k_indices]

            # Většinový hlas. np.bincount spočítá výskyty a np.argmax vrátí první index (nejnižší třídu) maxima.
            y_pred[i] = np.argmax(np.bincount(k_nearest_classes))

        return y_pred

    def save(self, path: str) -> None:
        """Ulozi nauceny model do souboru ``.npz`` (format NumPy).

        U kNN je "naucenym modelem" cela trenovaci mnozina, takze se
        ukladaji pole ``x_train_`` a ``y_train_`` spolu s hyperparametrem
        ``k``. To je zamerny kontrast s PCA z cviceni 05, kde ``save``
        ukladalo jen nekolik malych naucenych poli (prumer, vlastni cisla,
        vlastni vektory) a puvodni data uz nebyla potreba; kNN si naopak
        musi odnest uplne vsechna trenovaci data.

        Ulozeny model lze pozdeji nacist pres ``np.load(path)`` a z poli
        ``x_train``, ``y_train`` a ``k`` rekonstruovat klasifikator (metrika
        ``Distance`` se injektuje znovu az pri vytvoreni nove instance).

        Parametry
        ---------
        path : str
            Cesta k vystupnimu souboru. ``np.savez`` doplni priponu
            ``.npz`` automaticky, pokud v ceste chybi.
        """
        # assert  Ověřte, že model je nafitovan (self.x_train_ není None)
        assert self.x_train_ is not None and self.y_train_ is not None, "Model nelze uložit, protože není nafitován."
        #
        # Ulozte pres np.savez(path, ...) tato tri pole:
        #   x_train = self.x_train_
        #   y_train = self.y_train_
        #   k       = np.asarray(self.k)
        np.savez(path, x_train=self.x_train_, y_train=self.y_train_, k=np.asarray(self.k))

    # Zamerne NENI @classmethod (na rozdil od PCA.load v cviceni 05): kNN
    # potrebuje ke svemu behu injektovanou metriku Distance, ktera se do .npz
    # NEUKLADA. Model se proto obnovuje na uz existujici instanci vytvorene
    # s vlastni metrikou:  KNNClassifier(k, dist).load(path)  — hodnota ``k``
    # z konstruktoru se pak prepise hodnotou nactenou ze souboru.
    def load(self, path: str) -> KNNClassifier:
        """Nacte nauceny model z ``.npz`` souboru (format NumPy).

        Nacteny model lze pouzit k predikci na novych datech. Metoda
        ``load`` nacte pole ``x_train``, ``y_train`` a ``k`` a ulozi je
        do atributu ``self.x_train_``, ``self.y_train_`` a ``self.k``.

        Parametry
        ---------
        path : str
            Cesta k vstupnimu souboru. Musi existovat a byt validni
            ``.npz`` soubor ulozeny pres ``save``.

        Navratova hodnota
        -----------------
        KNNClassifier
            Tato instance (``self``), aby slo retezit
            ``.load(...).predict(...)``.
        """
        # assert  Ověřte, že soubor existuje a je validní .npz soubor
        import os
        assert os.path.exists(path) or os.path.exists(path + '.npz'), f"Soubor na cestě {path} neexistuje."

        with np.load(path) as data:
            assert 'x_train' in data and 'y_train' in data and 'k' in data, "Soubor .npz nemá platný formát (chybí klíče)."
            self.x_train_ = data['x_train']
            self.y_train_ = data['y_train']
            self.k = int(data['k'])

        return self
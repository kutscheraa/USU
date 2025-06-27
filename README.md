# Machine learning college class project
---

# 🧠 BioClinicalBERT pro Analýzu Dlouhých Českých Diagnóz

Tento projekt demonstruje použití předtrénovaného modelu **BioClinicalBERT** pro **analýzu sentimentu** a **vizualizaci embeddingů** u synteticky generovaných **českých lékařských diagnóz**.

## 🧪 Cíle projektu

* Přeložit dlouhé české lékařské zprávy do angličtiny.
* Získat embeddingy pomocí modelu `emilyalsentzer/Bio_ClinicalBERT`.
* Zobrazit embeddingy ve 2D prostoru pomocí PCA.
* Klasifikovat sentiment (pozitivní, negativní, neutrální) pomocí logistické regrese.
* Porovnat predikce s očekávanými štítky a vyhodnotit výkon modelu.
* Vytvořit numerické skóre sentimentu (+1, 0, -1).

## 🧰 Použité technologie

* 📦 `transformers`, `torch` – práce s BERT modelem
* 🧠 `scikit-learn` – PCA, LabelEncoder, klasifikace, metriky
* 📊 `matplotlib`, `pandas` – vizualizace a přehled výstupů
* 🌐 `deep-translator` – překlad českých textů do angličtiny

## 🧬 Struktura kódu

1. **Datová příprava**

   * Generování 20-větných odstavců pro každou třídu (pozitivní, negativní, neutrální).
   * Překlad pomocí `GoogleTranslator`.

2. **Embeddingy**

   * Použití BERTu pro získání vektorové reprezentace každého přeloženého textu.
   * Výběr CLS tokenu jako embedding.

3. **Vizualizace**

   * Redukce dimenzionality pomocí PCA.
   * Barevné rozlišení dle sentimentu.

4. **Klasifikace**

   * Trénink `LogisticRegression` na embeddingech.
   * Výstup klasifikační zprávy (`classification_report`).

5. **Skórování**

   * Převod tříd na numerické skóre: +1 (pozitivní), 0 (neutrální), -1 (negativní).
   * Uložení dat do tabulky (`pandas.DataFrame`).

## 📈 Ukázka výstupu

* Graf: PCA projekce embeddingů do 2D prostoru.
* Tabulka: původní české diagnózy, překlad, skóre, predikce.
* Klasifikační report s přesností, recall a F1 skóre.

## 📝 Poznámky

* Diagnózy jsou **synteticky generované** a neodpovídají reálným datům.
* Projekt slouží pouze k **testování použitelnosti BioBERTu** pro český lékařský jazyk.

## 📁 Spuštění

1. Nainstaluj závislosti:

   ```bash
   pip install transformers torch deep-translator scikit-learn
   ```

2. Spusť skript (`.ipynb` nebo `.py`) a sleduj výstupy.

---

!pip install transformers torch deep-translator scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

np.random.seed(42)

positive_sentences = [
    "Pacient se cítí dobře.", "Krevní tlak je v normě.", "Tělesná teplota je stabilní.",
    "EKG nevykazuje žádné abnormality.", "Laboratorní výsledky jsou uspokojivé.",
    "Bez známek zánětu.", "Pacient nemá žádné bolesti.", "Rána se hojí dobře.",
    "Dýchání je bez obtíží.", "Pacient bez teplot.", "Hmotnost je stabilní.",
    "Chuť k jídlu se zlepšila.", "Psychický stav je stabilní.", "Spánek je kvalitní.",
    "Bez edémů.", "Pacient spolupracuje.", "Mobilita je zachována.", "Bez neurologického deficitu.",
    "Výsledky jsou v referenčních mezích.", "Doporučena běžná kontrola."
]

negative_sentences = [
    "Pacient trpí silnými bolestmi.", "Laboratorní výsledky ukazují patologii.",
    "Zjištěna přítomnost metastáz.", "Stav vyžaduje intenzivní léčbu.",
    "Došlo ke zhoršení dýchání.", "Pacient má horečku.", "Zhoršená renální funkce.",
    "Chronické obtíže přetrvávají.", "Opakované zvracení.", "Pacient má edémy.",
    "Vysoký krevní tlak.", "CT ukázalo nález v mozku.", "Bolest se šíří do končetin.",
    "Zvýšený CRP.", "Závažná infekce v močových cestách.", "Neurologický deficit přetrvává.",
    "Nález na RTG je patologický.", "Pacient ztrácí na váze.", "Nutnost hospitalizace.",
    "Vyžaduje kontinuální monitoring."
]

neutral_sentences = [
    "Pacient je ve věku 68 let.", "Přijat na plánované vyšetření.",
    "V rodinné anamnéze diabetes.", "V současnosti užívá léky.",
    "Tlak měřen 3× denně.", "Kontrola za 2 týdny.", "Byl poučen o léčbě.",
    "Plánováno další vyšetření.", "RTG hrudníku provedeno.", "Doporučeno sledování.",
    "Výška 175 cm, váha 80 kg.", "Bez alergií.", "Byl proveden odběr krve.",
    "EKG naplánováno.", "Zatím bez změn.", "Anamnéza obsahuje prodělanou chřipku.",
    "Po dohodě s rodinou pokračuje léčba.", "Medikace zůstává beze změny.",
    "V ordinaci přítomen doprovod.", "Diagnóza potvrzena standardním postupem."
]

def generate_paragraphs(sentences, count):
    return [" ".join(np.random.choice(sentences, size=20, replace=True)) for _ in range(count)]

long_positive = generate_paragraphs(positive_sentences, 10)
long_negative = generate_paragraphs(negative_sentences, 10)
long_neutral = generate_paragraphs(neutral_sentences, 10)

cz_diagnoses = long_positive + long_negative + long_neutral
labels = ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 10

translated = [GoogleTranslator(source='auto', target='en').translate(text) for text in cz_diagnoses]

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

embeddings = [get_embedding(t) for t in translated]

pca = PCA(n_components=2)
points = pca.fit_transform(embeddings)

colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    plt.scatter(points[i, 0], points[i, 1], color=colors[label], label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title("Embeddingy dlouhých přeložených diagnóz (BioClinicalBERT)")
plt.legend()
plt.show()

le = LabelEncoder()
y = le.fit_transform(labels)
clf = LogisticRegression(max_iter=1000).fit(embeddings, y)
y_pred = clf.predict(embeddings)
print(classification_report(y, y_pred, target_names=le.classes_))

score_map = {"positive": 1, "neutral": 0, "negative": -1}
scored = [score_map[label] for label in labels]

pd.DataFrame({
    "Diagnóza (CZ)": cz_diagnoses,
    "Překlad (EN)": translated,
    "Skóre": scored,
    "Predikce": clf.predict(embeddings)
})
# %% Instalación e importación de librerías
%pip install gensim sklearn_crfsuite pandas numpy nltk
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import sklearn_crfsuite
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

# %% Importación del dataset
data = pd.read_csv("/data/SQLiV3.csv")  # Se asume que tiene columnas 'Sentence' y 'Label'
data = data[['Sentence', 'Label']]
data['Sentence'] = data['Sentence'].astype(str)
data = data.dropna(subset=['Sentence'])
data = data[data['Label'].isin(['0', '1'])]
print(data.head())

# %% Tokenización (sin eliminar caracteres especiales)
def preprocess_text(text):
    # Tokeniza preservando todo (los caracteres especiales se mantienen)
    tokens = word_tokenize(text)
    return tokens

data['Tokens'] = data['Sentence'].apply(preprocess_text)
sentences = data['Tokens'].tolist()

# Mostrar algunos ejemplos
for i, sentence in enumerate(sentences[:5]):
    print(f"Sentence {i}: {sentence}")

# %% Generación de embeddings con Word2Vec
embedding_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print("Vocabulario:", embedding_model.wv.index_to_key)

# %% Función para convertir una lista de tokens en un vector único (por pooling)
def sentence_to_average_embedding(tokens):
    # Obtener el embedding de cada token (si existe en el vocabulario)
    vectors = [embedding_model.wv[word] for word in tokens if word in embedding_model.wv]
    if vectors:
        # Se calcula el promedio (average pooling)
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(embedding_model.vector_size)

data['Avg_Embedding'] = data['Tokens'].apply(sentence_to_average_embedding)
print("Ejemplo de embedding promedio:")
print(data['Avg_Embedding'].head())

# %% Convertir el vector promedio en un diccionario de características
def embedding_to_feature_dict(embedding):
    return {f"dim_{i}": float(val) for i, val in enumerate(embedding)}

# Aquí, cada sentencia se representará como una secuencia de longitud 1 (una lista con un único dict)
data['Feature_Dict'] = data['Avg_Embedding'].apply(lambda emb: [embedding_to_feature_dict(emb)])
# La etiqueta global se coloca como una secuencia de longitud 1
data['Label_Seq'] = data['Label'].apply(lambda x: [x])

# %% Preparar datos para entrenamiento CRF
X = data['Feature_Dict'].tolist()   # Lista de secuencias; cada secuencia es una lista con un único dict
Y = data['Label_Seq'].tolist()        # Lista de secuencias de etiquetas (cada secuencia tiene un solo elemento)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print("Número de secuencias en entrenamiento:", len(X_train))
print("Número de secuencias en prueba:", len(X_test))

# %% Entrenar el modelo CRF
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, Y_train)

# %% Evaluar el modelo CRF
Y_pred = crf.predict(X_test)

# Aplanar las secuencias (aunque cada secuencia tiene 1 elemento)
Y_test_flat = [label for seq in Y_test for label in seq]
Y_pred_flat = [label for seq in Y_pred for label in seq]

print("Reporte de clasificación CRF:")
print(classification_report(Y_test_flat, Y_pred_flat))

# Mostrar matriz de confusión
cm = confusion_matrix(Y_test_flat, Y_pred_flat)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.show()

# Calcular la curva ROC y AUC (convertir etiquetas a enteros)
fpr, tpr, _ = roc_curve([int(x) for x in Y_test_flat], [int(x) for x in Y_pred_flat])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()

# %%
# Ejemplos de sentencias SQL basadas en el archivo SQLiV3.csv
test_sentences = [
    " OR 1=1 --", # SQLi
    "admin' --",  # SQLi
    "admin",  # SQLi
    "user", # Benigna
    "admin", # Benigna
    "password", # Benigna
    "asdasdasdasd", # Benigna
    "'", # SQLi
    "AND (SELECT count(tbl_name) FROM sqlite_master WHERE type='table' AND tbl_name NOT LIKE 'sqlite_%' ) < number_of_table ", # SQLi
    "SELECT * FROM users WHERE id = 1",  # Benigna
    "SELECT * FROM users WHERE id = 1 OR 1=1",  # Benigna
    "SELECT * FROM users WHERE username = 'admin' --",  # Benigna
    "SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1'",  # Benigna
    "SELECT * FROM users WHERE id = 1 OR pg_sleep(10) --",  # Benigna
    "INSERT INTO users (username, password) VALUES ('admin', 'password')",  # Benigna
    "UPDATE users SET password = 'newpassword' WHERE id = 1",  # Benigna
    "1; (load_file(char(47,101,116,99,47,112,97,115,115,119,100))) ,1,1,1;",  # SQLi
]

# Tokenizar las sentencias (preservando los caracteres especiales)
test_tokens = [word_tokenize(sentence) for sentence in test_sentences]

# Convertir cada sentencia en su embedding promedio usando la función ya definida: sentence_to_average_embedding
test_avg_embeddings = [sentence_to_average_embedding(tokens) for tokens in test_tokens]

# Convertir cada vector promedio en un diccionario de características.
# La función embedding_to_feature_dict toma un vector y lo transforma en un diccionario { 'dim_0': val0, 'dim_1': val1, ..., 'dim_99': val99 }
# Como el CRF espera una secuencia de diccionarios, envolvemos el resultado en una lista (cada sentencia se representa como una secuencia de longitud 1).
test_features = [[embedding_to_feature_dict(emb)] for emb in test_avg_embeddings]

# Predecir las etiquetas utilizando el modelo CRF entrenado
test_predictions = crf.predict(test_features)

# Mostrar los resultados
for sentence, prediction in zip(test_sentences, test_predictions):
    print(f"Sentence: {sentence}")
    print(f"Prediction: {prediction}")  # Cada predicción será una lista de longitud 1, ej. ['1'] o ['0']
    print()

# %%

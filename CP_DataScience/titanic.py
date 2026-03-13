import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


caminho_csv = os.path.join(os.path.dirname(__file__), "train.csv")
df = pd.read_csv(caminho_csv)

df["Sex"] = df["Sex"].map({
    "male": "Homem",
    "female": "Mulher"
})

df["Survived"] = df["Survived"].map({
    0: "Não Sobreviveu",
    1: "Sobreviveu"
})

# INFORMAÇÕES DA BASE

print("Primeiras linhas:")
print(df.head())

print("\nInformações da base:")
print(df.info())

print("\nPorcentagem de valores nulos:")
print(df.isnull().mean() * 100)

# LIMPEZA DE DADOS

df = df.drop(columns=["Cabin"])

df = df.dropna(subset=["Age"])

# ESTATÍSTICA DESCRITIVA

print("\nEstatísticas da Idade (Age)")
print("Média:", df["Age"].mean())
print("Mediana:", df["Age"].median())
print("Moda:", df["Age"].mode()[0])
print("Variância:", df["Age"].var())
print("Desvio padrão:", df["Age"].std())

print("\nEstatísticas do Preço do Ticket (Fare)")
print("Média:", df["Fare"].mean())
print("Mediana:", df["Fare"].median())
print("Moda:", df["Fare"].mode()[0])
print("Variância:", df["Fare"].var())
print("Desvio padrão:", df["Fare"].std())

# PORCENTAGEM DE SOBREVIVÊNCIA

print("\nPorcentagem de sobrevivência:")
print(df["Survived"].value_counts(normalize=True) * 100)

# PREÇO MÉDIO POR CLASSE

print("\nPreço médio do Ticket (£) por Classe:")
print(df.groupby("Pclass")["Fare"].mean())

# HISTOGRAMAS

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].hist(df["Age"].dropna(), bins="auto")
axes[0].set_title("Distribuição das Idades")
axes[0].set_xlabel("Idade")
axes[0].set_ylabel("Frequência")

axes[1].hist(df["Fare"], bins="auto")
axes[1].set_title("Distribuição do Preço do Ticket (£)")
axes[1].set_xlabel("Preço do Ticket (£)")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

# HEATMAP

corr = df[["Age","Fare","Pclass","SibSp","Parch"]].corr()

corr.index = ["Idade", "Preço do Ticket (£)", "Classe", "Irmãos/Cônjuge", "Pais/Filhos"]
corr.columns = ["Idade", "Preço do Ticket (£)", "Classe", "Irmãos/Cônjuge", "Pais/Filhos"]

plt.figure(figsize=(9,7))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Coeficiente de Correlação"}
)

plt.title("Mapa de Correlação entre Variáveis")

plt.xticks(rotation=30)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# SCATTERPLOTS

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].scatter(df["Age"], df["Fare"])
axes[0].set_title("Idade vs Preço do Ticket (£)")
axes[0].set_xlabel("Idade")
axes[0].set_ylabel("Preço do Ticket (£)")

axes[1].scatter(df["Pclass"], df["Fare"])
axes[1].set_title("Classe vs Preço do Ticket (£)")
axes[1].set_xlabel("Classe")
axes[1].set_ylabel("Preço do Ticket (£)")

plt.tight_layout()
plt.show()

# SOBREVIVÊNCIA

fig, axes = plt.subplots(1, 2, figsize=(12,5))

cores = {
    "Sobreviveu": "blue",
    "Não Sobreviveu": "red"
}

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df,
    hue_order=["Sobreviveu", "Não Sobreviveu"],
    palette=cores,
    ax=axes[0]
)

axes[0].set_title("Sobrevivência por Sexo")

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df,
    hue_order=["Sobreviveu", "Não Sobreviveu"],
    palette=cores,
    ax=axes[1]
)

axes[1].set_title("Sobrevivência por Classe")
axes[1].set_xlabel("Classe")
axes[1].set_ylabel("Numero de Passageiros")
axes[0].set_title("Sobrevivência por Sexo")
axes[0].set_xlabel("Sexo")
axes[0].set_ylabel("Número de Passageiros")
axes[0].legend(title="Situação")

plt.tight_layout()
plt.show()

# BOXPLOT COM E SEM OUTLIERS

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df_sem_outliers = df[(df["Fare"] >= limite_inferior) & (df["Fare"] <= limite_superior)]

fig, axes = plt.subplots(1, 2, figsize=(12,5))

axes[0].boxplot(df["Fare"].dropna())
axes[0].set_title("Preço do Ticket (£) COM Outliers")

axes[1].boxplot(df_sem_outliers["Fare"])
axes[1].set_title("Preço do Ticket (£) SEM Outliers")

plt.tight_layout()
plt.show()

# SCATTERPLOT COM E SEM OUTLIERS

df_scatter = df[["Age", "Fare"]].dropna()

Q1 = df_scatter["Age"].quantile(0.25)
Q3 = df_scatter["Age"].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df_sem_outliers = df_scatter[
    (df_scatter["Age"] >= limite_inferior) &
    (df_scatter["Age"] <= limite_superior)
]

fig, axes = plt.subplots(1, 2, figsize=(14,6))

axes[0].scatter(
    df_scatter["Age"],
    df_scatter["Fare"],
    color="blue",
    alpha=0.6,
    s=25
)

axes[0].set_title("Dados Originais com Outliers")
axes[0].set_xlabel("Idade")
axes[0].set_ylabel("Preço do Ticket (£)")

axes[1].scatter(
    df_sem_outliers["Age"],
    df_sem_outliers["Fare"],
    color="green",
    alpha=0.6,
    s=25
)

axes[1].set_title("Dados Após Remoção de Outliers")
axes[1].set_xlabel("Idade")
axes[1].set_ylabel("Preço do Ticket (£)")

plt.tight_layout()
plt.show()

# PREÇO DO TICKET POR CLASSE

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Pclass",
    y="Fare",
    data=df
)

plt.title("Preço do Ticket (£) por Classe")
plt.xlabel("Classe do Passageiro")
plt.ylabel("Preço do Ticket (£)")

plt.show()

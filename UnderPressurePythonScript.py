# %% [markdown]
# # Under Pressure - Análise e Previsão de Stress
# Pipeline completo de Machine Learning, desde a Análise Exploratória de Dados (EDA) até à otimização e avaliação de modelos.

# %% [markdown]
# ## 1. Importar as Bibliotecas Necessárias

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error,
)
import optuna
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve


# %% [markdown]
# ## 2. Carregamento de Dados

# %%
# Carregamento e Variáveis
df = pd.read_csv("archive/sleep_mobile_stress_dataset_15000.csv")

"""
user_id - Unique identifier assigned to each individual in the dataset;
age - Age of the individual in years.; gender - Gender of the individual (Male, Female, Other).
occupation - Occupation or professional background of the individual.
daily_screen_time_hours - Total number of hours spent using digital devices or smartphones per day.
phone_usage_before_sleep_minutes - Number of minutes spent using a smartphone before going to sleep.
sleep_duration_hours - Total hours of sleep the individual gets per night.
sleep_quality_score - A score representing sleep quality, typically ranging from 1 (poor) to 10 (excellent).
stress_level - Measured stress level of the individual on a scale from 1 (low stress) to 10 (high stress).
caffeine_intake_cups - Number of caffeinated beverages consumed per day.
physical_activity_minutes - Total minutes spent on physical activity or exercise per day.
notifications_received_per_day - Total number of smartphone notifications received in a day.
mental_fatigue_score - A score representing mental fatigue levels ranging from 1 (low fatigue) to 10 (high fatigue).
"""

# %% [markdown]
# ## 3. Análise Exploratória de Dados (EDA)

# %% [markdown]
# **Estatísticas dos dados**

# %%
print("Valores em falta:", df.isnull().sum().sum())
print("Valores duplicados:", df.duplicated().sum())

df = df.rename(
    columns={
        "notifications_received_per_day": "notifications_per_day",
        "phone_usage_before_sleep_minutes": "phone_before_sleep",
        "daily_screen_time_hours": "daily_screen_time",
        "physical_activity_minutes": "physical_activity",
    }
)

# Médias, Medianas, Variâncias
variables = [
    "stress_level",
    "mental_fatigue_score",
    "age",
    "daily_screen_time",
    "phone_before_sleep",
    "notifications_per_day",
    "sleep_duration_hours",
    "sleep_quality_score",
    "caffeine_intake_cups",
    "physical_activity",
]
stats_table = df[variables].agg(["mean", "median", "std", "var", "min", "max"]).T
stats_table = stats_table.round(3)
stats_table = stats_table.reset_index().rename(columns={"index": "Variável"})

# Criar a imagem da tabela
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")
tab = ax.table(
    cellText=stats_table.values,
    colLabels=stats_table.columns,
    cellLoc="center",
    loc="center",
)
tab.auto_set_font_size(False)
tab.set_fontsize(11)
tab.scale(1.5, 2.8)
plt.savefig("Tabela_Estatisticas_Geral.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
# Matriz de Correlação (Heatmap)
plt.figure(figsize=(14, 7))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.tight_layout()
plt.savefig("Matriz de Correlação (Heatmap).png", dpi=500)
plt.show()

# %% [markdown]
# **Moldura de Box Plots**

# %%

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

sns.boxplot(ax=axes[0], x="occupation", y="stress_level", data=df, palette="Set3", hue="occupation", legend=False)
axes[0].set_xlabel("Profissão")
axes[0].set_ylabel("Nível de Stress")
axes[0].tick_params(axis="x", rotation=45)

sns.boxplot(ax=axes[1], x="gender", y="stress_level", data=df, palette="Set3", hue="gender", legend=False)
axes[1].set_xlabel("Género")
axes[1].set_ylabel("Nível de Stress")

plt.tight_layout()
plt.savefig("Categorical_Stress.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Moldura de Scatter Plots**

# %%

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

sns.regplot(ax=axes[0, 0], x="daily_screen_time", y="stress_level", data=df, scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[0, 0].set_title("Tempo de Ecrã vs. Stress")
axes[0, 0].set_xlabel("Horas de Ecrã por Dia")
axes[0, 0].set_ylabel("Stress")

sns.regplot(ax=axes[0, 1], x="age", y="stress_level", data=df, scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[0, 1].set_title("Idade vs. Stress")
axes[0, 1].set_xlabel("Idade")
axes[0, 1].set_ylabel("Stress")

sns.regplot(ax=axes[1, 0], x="physical_activity", y="stress_level", data=df, scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[1, 0].set_title("Atividade Física vs. Stress")
axes[1, 0].set_xlabel("Minutos de Atividade Física")
axes[1, 0].set_ylabel("Stress")

sns.regplot(ax=axes[1, 1], x="sleep_quality_score", y="stress_level", data=df, scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
axes[1, 1].set_title("Qualidade do Sono vs. Stress")
axes[1, 1].set_xlabel("Índice de Qualidade de Sono")
axes[1, 1].set_ylabel("Stress")

plt.tight_layout()
plt.savefig("Dashboard_Stress.png", dpi=500)
plt.show()

# %% [markdown]
# ## 4. Pré-processamento e Criação das Variáveis Alvo

# %%
# One-Hot Encoding
X = df.drop(["user_id", "mental_fatigue_score", "stress_level"], axis=1)
X = pd.get_dummies(X, drop_first=True)

# Distribuição da Variável Alvo(Stress) - Criação de Y1, Y2, Y3
Y1 = np.where(df["stress_level"] < 5, 0, 1)

condicoes = [
    (df["stress_level"] < 2),
    (df["stress_level"] >= 2) & (df["stress_level"] < 4),
    (df["stress_level"] >= 4) & (df["stress_level"] < 6),
    (df["stress_level"] >= 6) & (df["stress_level"] < 8),
    (df["stress_level"] >= 8),
]
valores = [0, 1, 2, 3, 4]
Y2 = np.select(condicoes, valores, default=1)

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y3 = pd.cut(df["stress_level"], bins=bins, labels=False, include_lowest=True)

# %% [markdown]
# **Visualizar a Distribuição das classes do Stress**

# %%

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(22, 5))

# Histograma de 2 classes(0-1)
y1_counts = np.bincount(Y1)
sns.barplot(x=['No Stress (0)', 'Stress (1)'], y=y1_counts, ax=axes[0], palette=['green', 'red'])
axes[0].set_title('Cenario A - 2 classes(0-1)')
for i, v in enumerate(y1_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Histograma de 5 classes(0-4)
y2_counts = np.bincount(Y2)
colors_5 = ['green', 'blue', 'yellow', 'orange', 'red']
sns.barplot(x=np.arange(5), y=y2_counts, ax=axes[1], palette=colors_5)
axes[1].set_title('Cenario B - 5 classes(0-4)')
for i, v in enumerate(y2_counts):
    axes[1].text(i, v + 50, str(v), ha='center', fontsize=9)

# Histograma de 10 classes(0-9)
y3_counts = np.array([np.sum(Y3 == i) for i in range(10)])
colors_10 = sns.color_palette("RdYlGn_r", 10) 
sns.barplot(x=np.arange(10), y=y3_counts, ax=axes[2], palette=colors_10)
axes[2].set_title('Cenario C - 10 classes(0-9)')
axes[2].set_xlabel('Classes')
for i, v in enumerate(y3_counts):
    axes[2].text(i, v + 20, str(v), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("Distribuicao_Stess.png",dpi=500)
plt.show()

# %% [markdown]
# ## 5. Divisão de Treino/Teste e Scaling

# %%
scaler = StandardScaler()
smote = SMOTE(random_state=42)

X_train1, X_test1, Y1_train, Y1_test = train_test_split(
    X, Y1, test_size=0.25, random_state=42, stratify=Y1
)
X_train1_scaled = scaler.fit_transform(X_train1)
X_test1_scaled = scaler.transform(X_test1)


X_train2, X_test2, Y2_train, Y2_test = train_test_split(
    X, Y2, test_size=0.25, random_state=42, stratify=Y2
)
X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)
# SMOTE aplicado ao cenário Y2 (5 classes)
X_train2_resampled, Y2_train_resampled = smote.fit_resample(X_train2_scaled, Y2_train)

print("Distribuição original Y2:", np.bincount(Y2_train))
print("Distribuição após SMOTE Y2:", np.bincount(Y2_train_resampled))

X_train3, X_test3, Y3_train, Y3_test = train_test_split(
    X, Y3, test_size=0.25, random_state=42, stratify=Y3
)
X_train3_scaled = scaler.fit_transform(X_train3)
X_test3_scaled = scaler.transform(X_test3)
# SMOTE aplicado ao cenário Y3 (10 classes)
X_train3_resampled, Y3_train_resampled = smote.fit_resample(X_train3_scaled, Y3_train)

print("Distribuição original Y3:", np.array([np.sum(Y3_train == i) for i in range(10)]))
print("Distribuição após SMOTE Y3:", np.array([np.sum(Y3_train_resampled == i) for i in range(10)]))


# %% [markdown]
# ## 6. Modelos

# %% [markdown]
# ### **6.1 Redes Neuronais (Sklearn e PyTorch)**
# 
# Funções de Otimização e Gestão de Hardware

# %%
# --- NOTA: OTIMIZAÇÃO SKLEARN ---
def otimizar_mlp(X, y, nome_alvo, n_tentativas=200):
    print(f"\n>>> Otimizando para: {nome_alvo}")
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 2)
        hidden_layers = [trial.suggest_int(f'n_units_l{i}', 32, 128) for i in range(n_layers)]
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            alpha=trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            activation='relu',
            random_state=42,
            early_stopping=True
        )
        return cross_val_score(clf, X, y, cv=3, n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_tentativas)

    best = study.best_params
    camadas = [best[f'n_units_l{i}'] for i in range(best['n_layers'])]
    print("-" * 30)
    print(f"RESULTADOS PARA {nome_alvo}:")
    print(f"hidden_layer_sizes = {camadas}")
    print(f"alpha = {best['alpha']:.6f}")
    print(f"learning_rate_init = {best['learning_rate_init']:.6f}")
    print("-" * 30)
    return study.best_params

# --- NOTA: GESTÃO DE HARDWARE ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- NOTA: OTIMIZAÇÃO PYTORCH ---
def otimizar_torch(X_train, Y_train, X_test, Y_test, n_tentativas, num_classes=0,weights=None):
    print("\n>>> Otimizando PyTorch com Optuna")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    class_weights_tensor = torch.FloatTensor(weights).to(device) if weights is not None else None

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        camada_1 = trial.suggest_int("camada_1", 32, 256)
        camada_2 = trial.suggest_int("camada_2", 16, 128)
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)

        modelo = nn.Sequential(
            nn.Linear(X_train.shape[1], camada_1), nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(camada_1, camada_2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(camada_2, num_classes)
        ).to(device)

        otimizador = optim.Adam(modelo.parameters(), lr=lr)
        criterio = nn.CrossEntropyLoss(weight=class_weights_tensor)

        X_tensor = torch.FloatTensor(X_train)
        Y_tensor = torch.tensor(Y_train, dtype=torch.long)
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        modelo.train()
        for _ in range(20):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                otimizador.zero_grad()
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)
                loss.backward()
                otimizador.step()

        modelo.eval()
        with torch.no_grad():
            outputs = modelo(X_test_tensor)
            _, predicoes = torch.max(outputs, 1)
            acc = accuracy_score(Y_test, predicoes.cpu().numpy())
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_tentativas)
    print("-" * 30)
    print("MELHORES PARÂMETROS PYTORCH:")
    print(study.best_params)
    print("-" * 30)
    return study.best_params

# --- NOTA: TREINO FINAL PYTORCH ---
def treinar_rapido_torch(X_train, Y_train, epochs=50, batch_size=64,num_classes=0,n1=0,n2=0,dropout_rate=0.0,lr=0.001,weights=None):
    class_weights_tensor = torch.FloatTensor(weights).to(device) if weights is not None else None
    
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.tensor(Y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, Y_tensor) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    modelo = nn.Sequential(
        nn.Linear(X_train.shape[1], n1), nn.ReLU(),
        nn.Linear(n1, n2), nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(n2, num_classes)
    ).to(device)

    otimizador = optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss(weight=class_weights_tensor)

    for _ in range(epochs):
        modelo.train() 
        for inputs, labels in train_loader: 
            inputs, labels = inputs.to(device), labels.to(device) 
            otimizador.zero_grad()
            outputs = modelo(inputs) 
            loss = criterio(outputs, labels) 
            loss.backward() 
            otimizador.step() 
    return modelo

# %% [markdown]
# Otimização de parâmetros

# %% [markdown]
# Treino com os parâmetros otimizados e Avaliação final

# %%

clf_mlp_y1 = MLPClassifier(
    hidden_layer_sizes=[68],
    alpha=0.000090,
    learning_rate_init=0.001325,
    activation='relu',
    random_state=42,
    max_iter=1000,
    early_stopping=True
)

clf_mlp_y1.fit(X_train1_scaled, Y1_train)
y_pred_mlp_y1 = clf_mlp_y1.predict(X_test1_scaled)
print("\n--- AVALIAÇÃO FINAL MLPCLASSIFIER com Y1 ---")
print(classification_report(Y1_test, y_pred_mlp_y1))

# --- ALVO Y2: PyTorch ---
# --- Sem SMOTE ----
modelo_y2_torch = treinar_rapido_torch(
    X_train2_scaled, Y2_train, 
    epochs=100, 
    num_classes=5,
    n1=51,  
    n2=83, 
    dropout_rate=0.31414113671415267, 
    lr=0.00021771858227099064
)
modelo_y2_torch.eval() 
with torch.no_grad(): 
    X_test_tensor = torch.FloatTensor(X_test2_scaled).to(device)
    outputs = modelo_y2_torch(X_test_tensor) 
    _, predicoes_y2 = torch.max(outputs, 1) 

#--- Com SMOTE ---
# PyTorch Y2 com SMOTE
modelo_y2_smote = treinar_rapido_torch(
    X_train2_resampled, Y2_train_resampled, 
    epochs=100, 
    num_classes=5,
    n1=51,  
    n2=83, 
    dropout_rate=0.31414113671415267, 
    lr=0.00021771858227099064
)
modelo_y2_smote.eval()
with torch.no_grad():
    X_test2_tensor = torch.FloatTensor(X_test2_scaled).to(device)
    outputs = modelo_y2_smote(X_test2_tensor)
    _, predicoes_y2_smote = torch.max(outputs, 1)
    
print("\n--- AVALIAÇÃO FINAL PyTorch com Y2 ---")
print(classification_report(Y2_test, predicoes_y2.cpu().numpy()))
print(classification_report(Y2_test, predicoes_y2_smote.cpu().numpy()))

# --- ALVO Y3: PyTorch  ---

# --- Sem SMOTE ----
modelo_y3 = treinar_rapido_torch(X_train3_scaled, Y3_train, epochs=100, num_classes=10, n1=200, n2=106, dropout_rate=0.052526948320933936, lr=0.0002473651052415622)
modelo_y3.eval() 
with torch.no_grad(): 
    X_test_tensor = torch.FloatTensor(X_test3_scaled).to(device)
    outputs = modelo_y3(X_test_tensor) 
    _, predicoes_y3 = torch.max(outputs, 1) 
    
#--- Com SMOTE ---
modelo_y3_smote = treinar_rapido_torch(X_train3_resampled, Y3_train_resampled, epochs=100, num_classes=10,n1=200, n2=106, dropout_rate=0.052526948320933936, lr=0.0002473651052415622)
modelo_y3_smote.eval()
with torch.no_grad():
    X_test3_tensor = torch.FloatTensor(X_test3_scaled).to(device)
    outputs = modelo_y3_smote(X_test3_tensor)
    _, predicoes_y3_smote = torch.max(outputs, 1)  
       
print("\n--- AVALIAÇÃO MELHORADA PyTorch com Y3 ---")
print(classification_report(Y3_test, predicoes_y3.cpu().numpy()))
print(classification_report(Y3_test, predicoes_y3_smote.cpu().numpy() ))


# %% [markdown]
# ### **6.2: Support Vector Machines (SVM)**
# 
# Funções de Otimização

# %%

def objectiveSVM(trial,X_train,y_train):
    c = trial.suggest_float("C", 0.1, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    svc = SVC(C=c, kernel=kernel, class_weight='balanced', random_state=42)
    # Usamos cross-validation simples para ser rápido
    score = cross_val_score(svc, X_train, y_train, n_jobs=-1, cv=3).mean()
    return score

study_svm1 = optuna.create_study(direction="maximize")
study_svm1.optimize(lambda trial: objectiveSVM(trial,X_train1_scaled, Y1_train), n_trials=15)
print("Melhores parâmetros SVM para Y1:", study_svm1.best_params)

study_svm2 = optuna.create_study(direction="maximize")
study_svm2.optimize(lambda trial: objectiveSVM(trial,X_train2_scaled, Y2_train), n_trials=15)
print("Melhores parâmetros SVM para Y2:", study_svm2.best_params)

study_svm3 = optuna.create_study(direction="maximize")
study_svm3.optimize(lambda trial: objectiveSVM(trial,X_train3_scaled, Y3_train), n_trials=15)
print("Melhores parâmetros SVM para Y3:", study_svm3.best_params)


# %% [markdown]
# Execução SVM

# %%
svm1 = SVC(**study_svm1.best_params, probability=True, class_weight='balanced', random_state=42)
svm1.fit(X_train1_scaled, Y1_train)
SVM_Y1_pred=svm1.predict(X_test1_scaled)


svm2 = SVC(**study_svm2.best_params, probability=True, class_weight='balanced', random_state=42)
svm2.fit(X_train2_scaled, Y2_train)
SVM_Y2_pred=svm2.predict(X_test2_scaled)

svm2_smote = SVC(**study_svm2.best_params, probability=True, class_weight='balanced', random_state=42)
svm2_smote.fit(X_train2_resampled, Y2_train_resampled)
SVM_Y2_smote_pred = svm2_smote.predict(X_test2_scaled)


svm3 = SVC(**study_svm3.best_params, probability=True, class_weight='balanced', random_state=42)
svm3.fit(X_train3_scaled, Y3_train)
SVM_Y3_pred=svm3.predict(X_test3_scaled)

svm3_smote = SVC(**study_svm3.best_params, probability=True, class_weight='balanced', random_state=42)
svm3_smote.fit(X_train3_resampled, Y3_train_resampled)
SVM_Y3_smote_pred = svm3_smote.predict(X_test3_scaled)


print("=================== SVM ===================")
print("--- Y1 ---")
print(classification_report(Y1_test, SVM_Y1_pred,zero_division=0))
print("--- Y2 ---")
print(classification_report(Y2_test, SVM_Y2_pred,zero_division=0))
print("--- Y2 (SMOTE) ---")
print(classification_report(Y2_test, SVM_Y2_smote_pred,zero_division=0))
print("--- Y3 ---")
print(classification_report(Y3_test, SVM_Y3_pred,zero_division=0))
print("--- Y3 (SMOTE) ---")
print(classification_report(Y3_test, SVM_Y3_smote_pred,zero_division=0))

# %% [markdown]
# ### 6.3 **Random Forest**

# %% [markdown]
# Funções de Otimização

# %%
def objectiveRF(trial,X_train,y_train):
    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        n_jobs=-1, 
        random_state=42
    )
    score = cross_val_score(rf, X_train,y_train, n_jobs=-1, cv=3).mean()
    return score

study_rf1 = optuna.create_study(direction="maximize")
study_rf1.optimize(lambda trial: objectiveRF(trial,X_train1_scaled, Y1_train), n_trials=30)
print("Melhores parâmetros para RF com Y1:", study_rf1.best_params)

study_rf2 = optuna.create_study(direction="maximize")
study_rf2.optimize(lambda trial: objectiveRF(trial,X_train2_scaled, Y2_train), n_trials=30)
print("Melhores parâmetros para RF com Y2:", study_rf2.best_params)

study_rf3 = optuna.create_study(direction="maximize")
study_rf3.optimize(lambda trial: objectiveRF(trial,X_train3_scaled, Y3_train), n_trials=30)
print("Melhores parâmetros para RF com Y3:", study_rf3.best_params)


# %% [markdown]
# Execução e Avaliação

# %%
rf1 = RandomForestClassifier(**study_rf1.best_params,n_jobs=-1, random_state=42)
rf1.fit(X_train1_scaled, Y1_train)
RF_Y1_pred = rf1.predict(X_test1_scaled)

rf2 = RandomForestClassifier(**study_rf2.best_params, n_jobs=-1, random_state=42)
rf2.fit(X_train2_scaled, Y2_train)
RF_Y2_pred = rf2.predict(X_test2_scaled)

rf2_smote = RandomForestClassifier(**study_rf2.best_params, n_jobs=-1, random_state=42)
rf2_smote.fit(X_train2_resampled, Y2_train_resampled)
RF_Y2_smote_pred = rf2_smote.predict(X_test2_scaled)

rf3 = RandomForestClassifier(**study_rf3.best_params, n_jobs=-1, random_state=42)
rf3.fit(X_train3_scaled, Y3_train)
RF_Y3_pred = rf3.predict(X_test3_scaled)

rf3_smote = RandomForestClassifier(**study_rf3.best_params, n_jobs=-1, random_state=42)
rf3_smote.fit(X_train3_resampled, Y3_train_resampled)
RF_Y3_smote_pred = rf3_smote.predict(X_test3_scaled)


print("\n=================== RANDOM FOREST ===================")
print("--- Y1 ---")
print(classification_report(Y1_test, RF_Y1_pred,zero_division=0))
print("--- Y2 ---")
print(classification_report(Y2_test, RF_Y2_pred,zero_division=0))
print("--- Y2 (SMOTE) ---")
print(classification_report(Y2_test, RF_Y2_smote_pred,zero_division=0))
print("--- Y3 ---")
print(classification_report(Y3_test, RF_Y3_pred,zero_division=0))
print("--- Y3 (SMOTE) ---")
print(classification_report(Y3_test, RF_Y3_smote_pred,zero_division=0))

# %% [markdown]
# ###  6.4 **Naive-Bayes**

# %%
nb1 = GaussianNB()
nb2 = GaussianNB()
nb3 = GaussianNB()

nb1.fit(X_train1_scaled, Y1_train)
NB_Y1_pred=nb1.predict(X_test1_scaled)

nb2.fit(X_train2_scaled, Y2_train)
NB_Y2_pred=nb2.predict(X_test2_scaled)

nb2_smote = GaussianNB()
nb2_smote.fit(X_train2_resampled, Y2_train_resampled)
NB_Y2_smote_pred = nb2_smote.predict(X_test2_scaled)

nb3.fit(X_train3_scaled, Y3_train)
NB_Y3_pred=nb3.predict(X_test3_scaled)

nb3_smote = GaussianNB()
nb3_smote.fit(X_train3_resampled, Y3_train_resampled)
NB_Y3_smote_pred = nb3_smote.predict(X_test3_scaled)

print("\n=================== NAIVE BAYES ===================")
print("--- Y1 ---")
print(classification_report(Y1_test, NB_Y1_pred,zero_division=0))
print("--- Y2 ---")
print(classification_report(Y2_test, NB_Y2_pred,zero_division=0))
print("--- Y2 (SMOTE) ---")
print(classification_report(Y2_test, NB_Y2_smote_pred,zero_division=0))
print("--- Y3 ---")
print(classification_report(Y3_test, NB_Y3_pred,zero_division=0))
print("--- Y3 (SMOTE) ---")
print(classification_report(Y3_test, NB_Y3_smote_pred,zero_division=0))

# %% [markdown]
# ### 6.5 **Voting Classifier**

# %%
# Cenário Y1 (2 Classes)
vc1 = VotingClassifier(estimators=[('nn',clf_mlp_y1), ('svm', svm1), ('rf', rf1), ('nb', nb1)], voting='soft',n_jobs=-1)
vc1.fit(X_train1_scaled, Y1_train)
VC_Y1_pred = vc1.predict(X_test1_scaled)


# Cenário Y2 (5 Classes)
vc2 = VotingClassifier(estimators=[('svm', svm2), ('rf', rf2), ('nb', nb2)], voting='soft',n_jobs=-1)
vc2.fit(X_train2_scaled, Y2_train)
VC_Y2_pred = vc2.predict(X_test2_scaled)

vc2_smote = VotingClassifier(estimators=[('svm', svm2_smote), ('rf', rf2_smote), ('nb', nb2_smote)], voting='soft', n_jobs=-1)
vc2_smote.fit(X_train2_resampled, Y2_train_resampled)
VC_Y2_smote_pred = vc2_smote.predict(X_test2_scaled)


# Cenário Y3 (10 Classes)
vc3 = VotingClassifier(estimators=[('svm', svm3), ('rf', rf3), ('nb', nb3)], voting='soft',n_jobs=-1)
vc3.fit(X_train3_scaled, Y3_train)
VC_Y3_pred = vc3.predict(X_test3_scaled)

vc3_smote = VotingClassifier(estimators=[('svm', svm3_smote), ('rf', rf3_smote), ('nb', nb3_smote)], voting='soft', n_jobs=-1)
vc3_smote.fit(X_train3_resampled, Y3_train_resampled)
VC_Y3_smote_pred = vc3_smote.predict(X_test3_scaled)


print("\n=================== VOTING CLASSIFIER ===================")
print("--- Y1 ---")
print(classification_report(Y1_test, VC_Y1_pred,zero_division=0))
print("--- Y2 ---")
print(classification_report(Y2_test, VC_Y2_pred,zero_division=0))
print("--- Y2 (SMOTE) ---")
print(classification_report(Y2_test, VC_Y2_smote_pred,zero_division=0))
print("--- Y3 ---")
print(classification_report(Y3_test, VC_Y3_pred,zero_division=0))
print("--- Y3 (SMOTE) ---")
print(classification_report(Y3_test, VC_Y3_smote_pred,zero_division=0))

# %% [markdown]
# ## Gráficos

# %% [markdown]
# ### Matrizes de Confusão

# %% [markdown]
#  Primeiro Cenário 

# %%
fig = plt.figure(figsize=(15, 12)) 

cenarios1 = [
    (Y1_test, y_pred_mlp_y1, 'Cenário Y1 - MLPClassifier(2 Classes)'),
    (Y1_test, SVM_Y1_pred, 'Cenário Y1 - Support Vector Machine(2 Classes)'),
    (Y1_test, RF_Y1_pred, 'Cenário Y1 - Random Forest(2 Classes)'),
    (Y1_test, NB_Y1_pred, 'Cenário Y1 - Naive Bayes (2 Classes)'),
    (Y1_test, VC_Y1_pred, 'Cenário Y1 - Voting Classifier (2 Classes)'),
]

posicoes = [
    (0, 0), (0, 2),  # Linha 0 (Cima): Esquerda e Direita
    (1, 0), (1, 2),  # Linha 1 (Meio): Esquerda e Direita
    (2, 1)           # Linha 2 (Baixo): Centro!
]

# 3. No loop, criamos o eixo na posição correta antes de plotar
for pos, (y_true, y_pred, title) in zip(posicoes, cenarios1):
    # Cria o subplot dinamicamente na grade 3x4 ocupando 2 colunas de largura (colspan=2)
    ax = plt.subplot2grid((3, 4), pos, colspan=2)
    
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true') 
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel('Previsão do Modelo')
    ax.set_ylabel('Classe Real')

plt.tight_layout()
plt.savefig("Matizes_Confusao_Primeiro_Cenario.png", dpi=500)
plt.show()

# %% [markdown]
# ### Segundo Cenário

# %%
fig, axes = plt.subplots(5, 2, figsize=(25, 30))

cenarios2 = [
    (Y2_test, predicoes_y2.cpu().numpy(), 'Cenário Y2 - PyTorch(5 Classes)'),
    (Y2_test, predicoes_y2_smote.cpu().numpy(), 'Cenário Y2 - PyTorch (SMOTE) (5 Classes)'),

    (Y2_test, SVM_Y2_pred, 'Cenário Y2 - Support Vector Machine(5 Classes)'),
    (Y2_test, SVM_Y2_smote_pred, 'Cenário Y2 - Support Vector Machine (SMOTE) (5 Classes)'),

    (Y2_test, RF_Y2_pred, 'Cenário Y2 - Random Forest(5 Classes)'),
    (Y2_test, RF_Y2_smote_pred, 'Cenário Y2 - Random Forest (SMOTE) (5 Classes)'),

    (Y2_test, NB_Y2_pred, 'Cenário Y2 - Naive Bayes (5 Classes)'),
    (Y2_test, NB_Y2_smote_pred, 'Cenário Y2 - Naive Bayes (SMOTE) (5 Classes)'),
    
    (Y2_test, VC_Y2_pred, 'Cenário Y2 - Voting Classifier (5 Classes)'),
    (Y2_test, VC_Y2_smote_pred, 'Cenário Y2 - Voting Classifier (SMOTE) (5 Classes)')
    
]

for ax, (y_true, y_pred, title) in zip(axes.flatten(), cenarios2):
    # 'true' normaliza sobre as linhas (os valores reais)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true') 
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel('Previsão do Modelo')
    ax.set_ylabel('Classe Real')

plt.tight_layout()
plt.savefig("Matizes_Confusao_Segundo_Cenario.png", dpi=500)
plt.show()

# %% [markdown]
# ### Terceiro Cenário

# %%
fig, axes = plt.subplots(5, 2, figsize=(25, 30))

cenarios = [

    (Y3_test, predicoes_y3.cpu().numpy(), 'Cenário Y3 - PyTorch(10 Classes)'),
    (Y3_test, predicoes_y3_smote.cpu().numpy(), 'Cenário Y3 - PyTorch (SMOTE) (10 Classes)'),

    (Y3_test, SVM_Y3_pred, 'Cenário Y3 - Support Vector Machine(10 Classes)'),
    (Y3_test, SVM_Y3_smote_pred, 'Cenário Y3 - Support Vector Machine (SMOTE) (10 Classes)'),

    (Y3_test, RF_Y3_pred, 'Cenário Y3 - Random Forest(10 Classes)'),
    (Y3_test, RF_Y3_smote_pred, 'Cenário Y3 - Random Forest (SMOTE) (10 Classes)'),

    (Y3_test, NB_Y3_pred, 'Cenário Y3 - Naive Bayes (10 Classes)'),
    (Y3_test, NB_Y3_smote_pred, 'Cenário Y3 - Naive Bayes (SMOTE) (10 Classes)'),
    
    (Y3_test, VC_Y3_pred, 'Cenário Y3 - Voting Classifier (10 Classes)'),
    (Y3_test, VC_Y3_smote_pred, 'Cenário Y3 - Voting Classifier (SMOTE) (10 Classes)')
    
]

for ax, (y_true, y_pred, title) in zip(axes.flatten(), cenarios):
    # 'true' normaliza sobre as linhas (os valores reais)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true') 
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel('Previsão do Modelo')
    ax.set_ylabel('Classe Real')

plt.tight_layout()
plt.savefig("Matizes_Confusao_Terceiro_Cenario.png", dpi=500)
plt.show()

# %% [markdown]
# ### Comparação dos modelos

# %% [markdown]
# ### Curvas ROC e AUC para o primeiro Cenário

# %%
plt.figure(figsize=(8, 6))
plt.title('Curvas ROC - Cenário Y1 (Comparação de Modelos)', fontsize=14, fontweight='bold')

modelos_y1 = {
    'NN ': y_pred_mlp_y1,
    'SVM ': SVM_Y1_pred,
    'Naive Bayes': NB_Y1_pred,
    'Random Forest': RF_Y1_pred,
    'Voting Classifier': VC_Y1_pred
}

colors = ['purple', 'blue', 'green', 'red', 'orange']

for (nome, y_pred), color in zip(modelos_y1.items(), colors):
    fpr, tpr, _ = roc_curve(Y1_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{nome} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Linha de base
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Curvas_ROC_Y1.png", dpi=500)
plt.show()

# %% [markdown]
# ### Métricas Gerais

# %%
#Funções de métricas

def accuracy_tolerance_1(y_true, y_pred):
    # Calcula a diferença absoluta entre o real e o previsto
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    # Retorna a média de casos onde a diferença é 0 ou 1
    return np.mean(diff <= 1)

def accuracy_tolerance_2(y_true, y_pred):
    # Calcula a diferença absoluta entre o real e o previsto
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    # Retorna a média de casos onde a diferença é 0, 1 ou 2
    return np.mean(diff <= 2)


def calcular_metricas(y_true, y_pred, nome_modelo, cenario):
    return {
        "Modelo": nome_modelo,
        "Cenário": cenario,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Accuracy Tolerance 1": accuracy_tolerance_1(y_true, y_pred),
        "Accuracy Tolerance 2": accuracy_tolerance_2(y_true, y_pred),
    }

# %% [markdown]
# Primeiro Cenário

# %%

resultados1 = []
 
# Y1
for nome, pred in [("NN", y_pred_mlp_y1), ("SVM", SVM_Y1_pred), ("Random Forest", RF_Y1_pred), ("Naive Bayes", NB_Y1_pred),("VotingClassifier", VC_Y1_pred)]:
    resultados1.append(calcular_metricas(Y1_test, pred, nome, "Y1 (2 classes)"))
 
df_resultados1 = pd.DataFrame(resultados1)
 
# Gráfico de comparação — barras agrupadas por cenário
metricas = ["Accuracy", "F1-weighted", "F1-macro"]
cenarios = df_resultados1["Cenário"].unique()
modelos = df_resultados1["Modelo"].unique()
cores = {"NN": "purple", "SVM": "blue", "Random Forest": "green", "Naive Bayes": "red", "VotingClassifier": "orange"}
 
fig, axes = plt.subplots(len(cenarios), 1, figsize=(8, 5), sharey=False,squeeze=False)
 
for ax, cenario in zip(axes.flatten(), cenarios):
    df_c = df_resultados1[df_resultados1["Cenário"] == cenario].set_index("Modelo")
    x = np.arange(len(metricas))
    largura = 0.15
 
    for j, modelo in enumerate(modelos):
        valores = [df_c.loc[modelo, m] for m in metricas]
        ax.bar(x + j * largura, valores, largura, label=modelo,
               color=cores.get(modelo, None), alpha=0.85)
 
    ax.set_title(cenario, fontsize=13, fontweight="bold")
    ax.set_xticks(x+largura*(len(modelos)-1)/2)
    ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
 
    # Anotar valores nas barras
    for bar in ax.patches:
        altura = bar.get_height()
        if altura > 0:
            ax.annotate(f"{altura:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, altura),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
 
plt.suptitle("Comparação de Modelos - Cenário 1", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("Comparacao_Modelos - Cenário 1.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# Segundo Cenário

# %%
resultados2 = []

# Y2
for nome, pred in [("NN", predicoes_y2.cpu().numpy()), ("SVM", SVM_Y2_pred), ("Random Forest", RF_Y2_pred), ("Naive Bayes", NB_Y2_pred),("VotingClassifier", VC_Y2_pred)]:
    resultados2.append(calcular_metricas(Y2_test, pred, nome, "Y2 (5 classes)"))

for nome, pred in [("NN", predicoes_y2_smote.cpu().numpy()), ("SVM", SVM_Y2_smote_pred), ("Random Forest", RF_Y2_smote_pred),
                   ("Naive Bayes", NB_Y2_smote_pred), ("VotingClassifier", VC_Y2_smote_pred)]:
    resultados2.append(calcular_metricas(Y2_test, pred, nome, "Y2 SMOTE (5 classes)"))
    
    
df_resultados2 = pd.DataFrame(resultados2)

# Gráfico de comparação — barras agrupadas por cenário
metricas = ["Accuracy", "F1-weighted", "F1-macro","Accuracy Tolerance 1"]
cenarios = df_resultados2["Cenário"].unique()
modelos = df_resultados2["Modelo"].unique()
cores = {"NN": "purple", "SVM": "blue", "Random Forest": "green", "Naive Bayes": "red", "VotingClassifier": "orange"}
 
fig, axes = plt.subplots(len(cenarios), 1, figsize=(20, 10), sharey=False)
 
for ax, cenario in zip(axes, cenarios):
    df_c = df_resultados2[df_resultados2["Cenário"] == cenario].set_index("Modelo")
    x = np.arange(len(metricas))
    largura = 0.15
 
    for j, modelo in enumerate(modelos):
        valores = [df_c.loc[modelo, m] for m in metricas]
        ax.bar(x + j * largura, valores, largura, label=modelo,
               color=cores.get(modelo, None), alpha=0.85)
 
    ax.set_title(cenario, fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura * (len(modelos) - 1) / 2)
    ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
 
    # Anotar valores nas barras
    for bar in ax.patches:
        altura = bar.get_height()
        if altura > 0:
            ax.annotate(f"{altura:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, altura),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
 
plt.suptitle("Comparação de Modelos - Cenário 2", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("Comparacao_Modelos - Cenário 2.png", dpi=500, bbox_inches="tight")
plt.show()


# %% [markdown]
# MAE - Erro Médio Absoluto - 2º Cenário

# %%
plt.figure(figsize=(12, 6))
sns.barplot(data=df_resultados2, x="Modelo", y="MAE", hue="Cenário", palette="magma")

plt.title("Distância Média do Erro ", fontsize=14, fontweight="bold")
plt.ylabel("Erro Médio (em níveis de stress)")
plt.axhline(1.0, color='red', linestyle='--', label="Limite de 1 nível") # Linha de tolerância
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

# Anotações
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.tight_layout()
plt.savefig("Comparacao_Modelos_MAE_cenario_2.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# Terceiro Cenário

# %%
resultados3 = []

# Y3
for nome, pred in [("NN", predicoes_y3.cpu().numpy()), ("SVM", SVM_Y3_pred), ("Random Forest", RF_Y3_pred), ("Naive Bayes", NB_Y3_pred),("VotingClassifier", VC_Y3_pred)]:
    resultados3.append(calcular_metricas(Y3_test, pred, nome, "Y3 (10 classes)"))

for nome, pred in [("NN", predicoes_y3_smote.cpu().numpy()), ("SVM", SVM_Y3_smote_pred), ("Random Forest", RF_Y3_smote_pred),
                   ("Naive Bayes", NB_Y3_smote_pred), ("VotingClassifier", VC_Y3_smote_pred)]:
    resultados3.append(calcular_metricas(Y3_test, pred, nome, "Y3 SMOTE (10 classes)"))
    

df_resultados3 = pd.DataFrame(resultados3)

# Gráfico de comparação — barras agrupadas por cenário
metricas = ["Accuracy", "F1-weighted", "F1-macro","Accuracy Tolerance 1","Accuracy Tolerance 2"]
cenarios = df_resultados3["Cenário"].unique()
modelos = df_resultados3["Modelo"].unique()
cores = {"NN": "purple", "SVM": "blue", "Random Forest": "green", "Naive Bayes": "red", "VotingClassifier": "orange"}
 
fig, axes = plt.subplots(len(cenarios), 1, figsize=(16, 7), sharey=False)
 
for ax, cenario in zip(axes, cenarios):
    df_c = df_resultados3[df_resultados3["Cenário"] == cenario].set_index("Modelo")
    x = np.arange(len(metricas))
    largura = 0.15
 
    for j, modelo in enumerate(modelos):
        valores = [df_c.loc[modelo, m] for m in metricas]
        ax.bar(x + j * largura, valores, largura, label=modelo,
               color=cores.get(modelo, None), alpha=0.85)
 
    ax.set_title(cenario, fontsize=13, fontweight="bold")
    ax.set_xticks(x + largura * (len(modelos) - 1) / 2)
    ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis="y", linestyle="--", alpha=0.4)
 
    # Anotar valores nas barras
    for bar in ax.patches:
        altura = bar.get_height()
        if altura > 0:
            ax.annotate(f"{altura:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, altura),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
 
plt.suptitle("Comparação de Modelos - Cenário 3", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("Comparacao_Modelos - Cenário 3.png", dpi=500, bbox_inches="tight")
plt.show()


# %% [markdown]
# MAE - Erro Médio Absoluto - 3º Cenário

# %%
plt.figure(figsize=(12, 6))
sns.barplot(data=df_resultados3, x="Modelo", y="MAE", hue="Cenário", palette="magma")

plt.title("Distância Média do Erro ", fontsize=14, fontweight="bold")
plt.ylabel("Erro Médio (em níveis de stress)")
plt.axhline(1.0, color='red', linestyle='--', label="Limite de 1 nível") # Linha de tolerância
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

# Anotações
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                       textcoords='offset points')
plt.tight_layout()
plt.savefig("Comparacao_Modelos_MAE_cenario_3.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Métricas Individuais por classe

# %%
#Função para extrair as métricas de cada modelo e cenário

def obter_metricas_por_classe(y_true, y_pred, modelo_nome):
    # output_dict=True transforma o relatório em dicionário
    report = classification_report(y_true, y_pred, output_dict=True)
    
    dados = []
    # Iterar sobre as classes (ignorando 'accuracy', 'macro avg', etc.)
    for classe, metricas in report.items():
        if classe.isdigit(): # Garante que estamos a ver as classes (0, 1, 2...)
            dados.append({
                "Modelo": modelo_nome,
                "Classe": int(classe),
                "Precision": metricas["precision"],
                "Recall": metricas["recall"],
                "F1-Score": metricas["f1-score"]
            })
    return dados

# %% [markdown]
# Primeiro Cenário

# %%
dados_linhas1 = []
    
for nome, pred in  [("NN", y_pred_mlp_y1), ("SVM", SVM_Y1_pred), ("Random Forest", RF_Y1_pred), 
                    ("Naive Bayes", NB_Y1_pred), ("VotingClassifier", VC_Y1_pred)]:
    dados_linhas1.extend(obter_metricas_por_classe(Y1_test, pred, nome))

    
df_linhas1 = pd.DataFrame(dados_linhas1)

df_plot = df_linhas1.melt(id_vars=["Modelo", "Classe"], 
                         value_vars=["Precision", "Recall", "F1-Score"], 
                         var_name="Metrica", 
                         value_name="Valor")

# Definir as cores para manter a consistência com os teus gráficos anteriores
cores = {"NN": "purple", "SVM": "blue", "Random Forest": "green", 
         "Naive Bayes": "red", "VotingClassifier": "orange"}

# Criar a grelha de subplots (1 linha, 3 colunas)
g = sns.FacetGrid(df_plot, col="Metrica", hue="Modelo", height=5, aspect=1.2, 
                  palette=cores, hue_kws={'linestyle': ['-', '--']*5}) 
                  # O linestyle alterna entre contínuo (Normal) e tracejado (SMOTE) se os nomes forem parecidos

# Mapear o gráfico de linha em cada subplot
g.map(sns.lineplot, "Classe", "Valor", marker="o", linewidth=2)

# Ajustes de layout e legendas
g.set_axis_labels("Nível de Stress (Classe)", "Score")
g.set_titles("{col_name}", size=14, fontweight='bold')
g.add_legend(title="Modelos e Variantes", bbox_to_anchor=(1, 0.5))

# Refinar os eixos
for ax in g.axes.flat:
    ax.set_xticks(range(0, 2)) # Para o cenário de 2 classes (Y1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

plt.subplots_adjust(top=0.8)

plt.savefig("Comparacao_Modelos_Precision_Recall_F1_Y1.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# Segundo Cenário

# %%
cores_base = {
    "NN": "purple", 
    "SVM": "blue", 
    "Random Forest": "green", 
    "Naive Bayes": "red", 
    "VotingClassifier": "orange"
}

mapa_cores_completo = {}
for nome, cor in cores_base.items():
    mapa_cores_completo[nome] = cor
    mapa_cores_completo[f"{nome} + SMOTE"] = cor

# --- (O código de coleta de dados permanece igual) ---
dados_linhas2 = []
for nome, pred in [("NN", predicoes_y2.cpu().numpy()), ("SVM", SVM_Y2_pred), 
                   ("Random Forest", RF_Y2_pred), ("Naive Bayes", NB_Y2_pred), 
                   ("VotingClassifier", VC_Y2_pred)]:
    metricas = obter_metricas_por_classe(Y2_test, pred, nome)
    for m in metricas: m['Tipo'] = 'Original'
    dados_linhas2.extend(metricas)

for nome, pred in [("NN", predicoes_y2_smote.cpu().numpy()), ("SVM", SVM_Y2_smote_pred), 
                   ("Random Forest", RF_Y2_smote_pred), ("Naive Bayes", NB_Y2_smote_pred), 
                   ("VotingClassifier", VC_Y2_smote_pred)]:
    nome_smote = f"{nome} + SMOTE"
    metricas = obter_metricas_por_classe(Y2_test, pred, nome_smote)
    for m in metricas: m['Tipo'] = 'SMOTE'
    dados_linhas2.extend(metricas)

df_linhas2 = pd.DataFrame(dados_linhas2)
df_plot = df_linhas2.melt(id_vars=["Modelo", "Classe", "Tipo"], 
                          value_vars=["Precision", "Recall", "F1-Score"], 
                          var_name="Metrica", value_name="Valor")

# 3. Gerar o gráfico usando o palette expandido
g = sns.FacetGrid(df_plot, col="Metrica", row="Tipo", hue="Modelo", 
                  height=4, aspect=1.3, 
                  palette=mapa_cores_completo, # <--- Aplica as cores consistentes
                  sharey=True)

g.map(sns.lineplot, "Classe", "Valor", marker="o", linewidth=2)

# Ajustes estéticos finais
g.set_axis_labels("Nível de Stress (Classe)", "Score")
g.set_titles(row_template="{row_name}", col_template="{col_name}", size=12, fontweight='bold')

for ax in g.axes.flat:
    ax.set_xticks(range(0, 5)) # Para o cenário de 5 classes (Y2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

g.add_legend(title="Modelos", adjust_subtitles=True)

plt.subplots_adjust(top=0.9, hspace=0.3)
g.fig.suptitle("Comparação de Desempenho: Original vs SMOTE (Cores Consistentes)", 
               fontsize=16, fontweight='bold')

plt.savefig("Comparacao_Modelos_Precision_Recall_F1_Y2.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# Terceiro Cenário

# %%
cores_base = {
    "NN": "purple", 
    "SVM": "blue", 
    "Random Forest": "green", 
    "Naive Bayes": "red", 
    "VotingClassifier": "orange"
}

mapa_cores_completo = {}
for nome, cor in cores_base.items():
    mapa_cores_completo[nome] = cor
    mapa_cores_completo[f"{nome} + SMOTE"] = cor

# --- (O código de coleta de dados permanece igual) ---
dados_linhas3 = []
for nome, pred in [("NN", predicoes_y3.cpu().numpy()), ("SVM", SVM_Y3_pred), 
                   ("Random Forest", RF_Y3_pred), ("Naive Bayes", NB_Y3_pred), 
                   ("VotingClassifier", VC_Y3_pred)]:
    metricas = obter_metricas_por_classe(Y3_test, pred, nome)
    for m in metricas: m['Tipo'] = 'Original'
    dados_linhas3.extend(metricas)

for nome, pred in [("NN", predicoes_y3_smote.cpu().numpy()), ("SVM", SVM_Y3_smote_pred), 
                   ("Random Forest", RF_Y3_smote_pred), ("Naive Bayes", NB_Y3_smote_pred), 
                   ("VotingClassifier", VC_Y3_smote_pred)]:
    nome_smote = f"{nome} + SMOTE"
    metricas = obter_metricas_por_classe(Y3_test, pred, nome_smote)
    for m in metricas: m['Tipo'] = 'SMOTE'
    dados_linhas3.extend(metricas)

df_linhas3 = pd.DataFrame(dados_linhas3)
df_plot = df_linhas3.melt(id_vars=["Modelo", "Classe", "Tipo"], 
                          value_vars=["Precision", "Recall", "F1-Score"], 
                          var_name="Metrica", value_name="Valor")

# 3. Gerar o gráfico usando o palette expandido
g = sns.FacetGrid(df_plot, col="Metrica", row="Tipo", hue="Modelo", 
                  height=4, aspect=1.3, 
                  palette=mapa_cores_completo, # <--- Aplica as cores consistentes
                  sharey=True)

g.map(sns.lineplot, "Classe", "Valor", marker="o", linewidth=2)

# Ajustes estéticos finais
g.set_axis_labels("Nível de Stress (Classe)", "Score")
g.set_titles(row_template="{row_name}", col_template="{col_name}", size=12, fontweight='bold')

for ax in g.axes.flat:
    ax.set_xticks(range(0, 10))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

g.add_legend(title="Modelos", adjust_subtitles=True)

plt.subplots_adjust(top=0.9, hspace=0.3)
g.fig.suptitle("Comparação de Desempenho: Original vs SMOTE (Cores Consistentes)", 
               fontsize=16, fontweight='bold')

plt.savefig("Comparacao_Modelos_Precision_Recall_F1_Y3.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Importância das variáveis

# %% [markdown]
# Random Forest

# %%
rf_models = [rf1, rf2, rf3] # Teus modelos já treinados
titulos = ['Cenário A', 'Cenário B ', 'Cenário C']
features = X.columns

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, model in enumerate(rf_models):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    axes[i].barh(range(len(indices)), importances[indices], color='seagreen')
    axes[i].set_yticks(range(len(indices)))
    axes[i].set_yticklabels([features[j] for j in indices])
    axes[i].set_title(f'RF Importance - {titulos[i]}')

plt.tight_layout()
plt.savefig("Feature_Importance_RF.png", dpi=500, bbox_inches="tight")
plt.show()

# %% [markdown]
# Curvas de aprendizagem

# %%

CORES_MODELOS = {
    "NN": "purple", 
    "SVM": "blue", 
    "Random Forest": "green", 
    "Naive Bayes": "red", 
    "VotingClassifier": "orange"
}


def plot_learning_curve(estimator, X, y, ax, titulo, cor, cv=5, n_pontos=8):
    """
    Plota a learning curve de um estimador num eixo dado.
    Mostra treino vs. validação com banda de desvio padrão.
    """
    train_sizes = np.linspace(0.1, 1.0, n_pontos)

    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring="accuracy",
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # Linha de treino
    ax.plot(sizes, train_mean, "o-", color=cor, linewidth=2, label="Treino")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.12, color=cor)

    # Linha de validação
    ax.plot(sizes, val_mean, "s--", color=cor, linewidth=2, alpha=0.7, label="Validação (CV)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.08, color=cor)

    # Gap final (diferença entre treino e validação no final)
    gap = train_mean[-1] - val_mean[-1]
    estado = "Overfitting" if gap > 0.05 else ("Underfitting" if val_mean[-1] < 0.55 else "OK")
    cor_estado = "darkred" if estado == "Overfitting" else ("darkorange" if estado == "Underfitting" else "darkgreen")

    ax.text(
        0.98, 0.05,
        f"Gap final: {gap:.3f} → {estado}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, color=cor_estado, fontweight="bold"
    )

    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Amostras de Treino")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(linestyle="--", alpha=0.3)



modelos_lc = [
    # (estimador,  X_scaled,       y,       cor,                      nome)
    (rf1,          X_train1_scaled, Y1_train, CORES_MODELOS["Random Forest"], "RF — Y1"),
    (svm1,         X_train1_scaled, Y1_train, CORES_MODELOS["SVM"],           "SVM — Y1"),
    (nb1,          X_train1_scaled, Y1_train, CORES_MODELOS["Naive Bayes"],   "NB — Y1"),

    (rf2,          X_train2_scaled, Y2_train, CORES_MODELOS["Random Forest"], "RF — Y2"),
    (svm2,         X_train2_scaled, Y2_train, CORES_MODELOS["SVM"],           "SVM — Y2"),
    (nb2,          X_train2_scaled, Y2_train, CORES_MODELOS["Naive Bayes"],   "NB — Y2"),

    (rf3,          X_train3_scaled, Y3_train, CORES_MODELOS["Random Forest"], "RF — Y3"),
    (svm3,         X_train3_scaled, Y3_train, CORES_MODELOS["SVM"],           "SVM — Y3"),
    (nb3,          X_train3_scaled, Y3_train, CORES_MODELOS["Naive Bayes"],   "NB — Y3"),
]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Learning Curves — Treino vs. Validação por Modelo e Cenário", fontsize=16, fontweight="bold")

for ax, (estimador, X_tr, y_tr, cor, titulo) in zip(axes.flatten(), modelos_lc):
    plot_learning_curve(estimador, X_tr, y_tr, ax, titulo, cor, cv=5, n_pontos=8)

plt.tight_layout()
plt.savefig("Learning_Curves.png", dpi=500, bbox_inches="tight")
plt.show()
print("Learning curves guardadas.")



#Codigo para realizar las ejecuciones

#Dual sin kernel, recordar que K = A*A^T, donde A es la matriz que tiene las xi en filas

#asi que tanto con Kernel com sin kernel K tiene dimensiones m * m

#Codigo para realizar las ejecuciones

#Dual sin kernel, recordar que K = A*A^T, donde A es la matriz que tiene las xi en filas

#asi que tanto con Kernel com sin kernel K tiene dimensiones m * m
import pandas as pd
import numpy as np
import tools as tl

# Configuración general de los experimentos solicitados
valores_nu = [0.1, 1.0, 10.0, 100.0]
semilla = 77214914
resultados = []

# ==========================================
# Dataset del Generador
# ==========================================
print("--- Iniciando Experimentos con Dataset del Generador ---")
# Usamos tu función generator_preprocess pasándole la dimensión x=4 del enunciado
X_gen_raw, y_gen_raw = tl.generator_preprocess("points_1000.dat", dim_x=4)
X_gen, y_gen = np.array(X_gen_raw), np.array(y_gen_raw)

for nu in valores_nu:
    X_train, X_test, y_train, y_test = tl.split_dataset(X_gen, y_gen, test_size=0.3, seed=semilla)
    
    # Resolver Primal Lineal (Convertimos a lista para respetar el tipado estricto)
    w_p, gam_p, _, obj_p, time_p = tl.primal_solve(X_train.tolist(), y_train.tolist(), "primal_svm.mod", nu)
    acc_p = tl.calculate_accuracy_linear(X_test, y_test, w_p, gam_p)
    
    # Resolver Dual Lineal (Matriz de producto escalar K = X * X^T)
    K_lineal_train = np.dot(X_train, X_train.T)
    lamb_l, obj_l, time_l = tl.dual_solve(K_lineal_train, y_train.tolist(), "dual_svm.mod", nu)
    
    # Recuperar plano y verificar que coincide con el Primal
    w_rec, gam_rec = tl.recover_linear_hyperplane(X_train, y_train, lamb_l, nu)
    acc_l = tl.calculate_accuracy_linear(X_test, y_test, w_rec, gam_rec)
    
    resultados.append({
        "Dataset": "Generador (Sintético)", "Nu": nu, "Semilla": seed, "Modelo": "Primal Lineal",
        "Funcion_Objetivo": obj_p, "Tiempo": time_p, "Accuracy": acc_p
    })
    resultados.append({
        "Dataset": "Generador (Sintético)", "Nu": nu, "Semilla": seed, "Modelo": "Dual Lineal",
        "Funcion_Objetivo": obj_l, "Tiempo": time_l, "Accuracy": acc_l
    })

# ==========================================
# Dataset Real (WDBC)
# ==========================================
print("\n--- Iniciando Experimentos con Dataset Real WDBC ---")
X_wdbc, y_wdbc = tl.process_wdbc("wdbc.data")

for nu in valores_nu:
    X_train, X_test, y_train, y_test = tl.split_dataset(X_wdbc, y_wdbc, test_size=0.3, seed=semilla)
    
    # Primal Lineal
    w_p, gam_p, _, obj_p, time_p = tl.primal_solve(X_train.tolist(), y_train.tolist(), "primal_svm.mod", nu)
    acc_p = tl.calculate_accuracy_linear(X_test, y_test, w_p, gam_p)
    
    # Dual Lineal 
    K_lineal_train = np.dot(X_train, X_train.T)
    lamb_l, obj_l, time_l = tl.dual_solve(K_lineal_train, y_train.tolist(), "dual_svm.mod", nu)
    w_rec, gam_rec = tl.recover_linear_hyperplane(X_train, y_train, lamb_l, nu)
    acc_l = tl.calculate_accuracy_linear(X_test, y_test, w_rec, gam_rec)
    
    # Dual con Kernel Gaussiano RBF (No Lineal)
    # Calculamos la matriz de Kernel RBF para Train usando tu función
    K_rbf_train = tl.compute_gaussian_kernel(X_train, X_train, sigma_sq=None)
    lamb_rbf, obj_rbf, time_rbf = tl.dual_solve(K_rbf_train, y_train.tolist(), "dual_svm.mod", nu)
    acc_rbf = tl.calculate_accuracy_rbf(X_train, y_train, X_test, y_test, lamb_rbf, nu, sigma_sq=None)
    
    resultados.append({
        "Dataset": "WDBC (Real)", "Nu": nu, "Semilla": seed, "Modelo": "Primal Lineal",
        "Funcion_Objetivo": obj_p, "Tiempo": time_p, "Accuracy": acc_p
    })
    resultados.append({
        "Dataset": "WDBC (Real)", "Nu": nu, "Semilla": seed, "Modelo": "Dual Lineal",
        "Funcion_Objetivo": obj_l, "Tiempo": time_l, "Accuracy": acc_l
    })
    resultados.append({
        "Dataset": "WDBC (Real)", "Nu": nu, "Semilla": seed, "Modelo": "Dual RBF (Gaussiano)",
        "Funcion_Objetivo": obj_rbf, "Tiempo": time_rbf, "Accuracy": acc_rbf
    })

# ==========================================
# Dataset NO SEPARABLE Linealmente (Moons / Círculos)
# ==========================================
print("\n--- Iniciando Experimentos con Dataset NO SEPARABLE ---")
from sklearn.datasets import make_moons

# Generamos 1000 puntos en forma de dos lunas entrelazadas
X_ns_raw, y_ns_raw = make_moons(n_samples=1000, noise=0.15, random_state=77214914)
y_ns_raw = np.where(y_ns_raw == 0, -1.0, 1.0)

for nu in valores_nu:
    X_train, X_test, y_train, y_test = tl.split_dataset(X_ns_raw, y_ns_raw, test_size=0.3, seed=semilla)
    
    # Intentar resolver con DUAL LINEAL (Sin Kernel)
    K_lineal_train = np.dot(X_train, X_train.T)
    lamb_l, obj_l, time_l = tl.dual_solve(K_lineal_train, y_train.tolist(), "dual_svm.mod", nu)
    
    # Recuperar el plano lineal para calcular la accuracy lineal
    w_rec_l, gam_rec_l = tl.recover_linear_hyperplane(X_train, y_train, lamb_l, nu)
    acc_l = tl.calculate_accuracy_linear(X_test, y_test, w_rec_l, gam_rec_l)
    
    # Resolver con DUAL RBF (Con Kernel Gaussiano)
    K_rbf_train = tl.compute_gaussian_kernel(X_train, X_train, sigma_sq=None)
    lamb_rbf, obj_rbf, time_rbf = tl.dual_solve(K_rbf_train, y_train.tolist(), "dual_svm.mod", nu)
    
    # Pasamos lamb_rbf directamente a calculate_accuracy_rbf, que clasifica usando únicamente la matriz de Kernel.
    acc_rbf = tl.calculate_accuracy_rbf(X_train, y_train, X_test, y_test, lamb_rbf, nu, sigma_sq=None)
    
    # Guardar en la estructura global de resultados
    resultados.append({
        "Dataset": "No Separable (Moons)", "Nu": nu, "Modelo": "Dual Lineal (Sin Kernel)",
        "Funcion_Objetivo": obj_l, "Tiempo": time_l, "Accuracy": acc_l
    })
    resultados.append({
        "Dataset": "No Separable (Moons)", "Nu": nu, "Modelo": "Dual RBF (Con Kernel)",
        "Funcion_Objetivo": obj_rbf, "Tiempo": time_rbf, "Accuracy": acc_rbf
    })

# ==========================================
# 3. Procesamiento y Exportación de Resultados (Corregido para 1 Semilla)
# ==========================================
df_resumen = pd.DataFrame(resultados)

# Como usas una única semilla, quitamos la columna 'Semilla' si existiera y guardamos los datos directamente.
# Así evitamos el .agg(["mean", "std"]) que generaba columnas llenas de NaNs/vacías.
if "Semilla" in df_resumen.columns:
    df_resumen = df_resumen.drop(columns=["Semilla"])

# Guardamos el archivo CSV final listo para la memoria
df_resumen.to_csv("reporte_final_svm.csv", index=False)
print("\n¡Todos los experimentos completados con éxito!")
print("Resultados consolidados guardados en 'reporte_final_svm.csv'.")

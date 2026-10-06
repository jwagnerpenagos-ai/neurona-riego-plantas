import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

print("Estamos creando una neurona para decidir si una planta necesita riego")
print("Salida 1 = regar | Salida 0 = no regar")

# Cada fila contiene dos entradas: [humedad del suelo %, temperatura °C]
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

print("Entradas \n", X)
print("Respuestas esperadas: \n", y.ravel().astype(int))


escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, mostrar=False):
    rng = np.random.default_rng(7)
    pesos = rng.normal(size=(2, 1))  
    sesgo = 0.0

    for epoca in range(epocas):
        z = X_normalizado @ pesos + sesgo
        predicciones = sigmoide(z)

        error = predicciones - y
        gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X_normalizado)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and epoca % 2000 == 0:
            perdida = np.mean(error ** 2)
            print(f"Epoca {epoca:5} | error: {perdida:.4f}")

    error_final = np.mean((sigmoide(X_normalizado @ pesos + sesgo) - y) ** 2)
    return pesos, sesgo, error_final


def probabilidad(datos, pesos, sesgo):
    return sigmoide((datos / escala) @ pesos + sesgo)


# ---------------------------------------------------------------
# Configuración inicial
# ---------------------------------------------------------------
tasa_aprendizaje = 0.5
epocas = 10000

print(f"\nEntrenamiento base: tasa = {tasa_aprendizaje}, epocas = {epocas}")
pesos, sesgo, error_final = entrenar(tasa_aprendizaje, epocas, mostrar=True)

print("\nPeso de la humedad:    ", round(pesos[0, 0], 4))
print("Peso de la temperatura:", round(pesos[1, 0], 4))
print("Sesgo aprendido:       ", round(sesgo, 4))
print("Error final:           ", round(error_final, 6))

print("\nSigno de los pesos:")
print("- Humedad negativo: a mayor humedad, menor probabilidad de regar."
      if pesos[0, 0] < 0 else
      "- Humedad positivo: a mayor humedad, mayor probabilidad de regar.")
print("- Temperatura positivo: a mayor temperatura, mayor probabilidad de regar."
      if pesos[1, 0] > 0 else
      "- Temperatura negativo: a mayor temperatura, menor probabilidad de regar.")

print("\nProbamos lo aprendido: ")
print("Si la salida es >= 0.5 = 1")
print("Si es < 0.5 = 0")

probabilidades = probabilidad(X, pesos, sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

for entrada, prob, respuesta, esperado in zip(X, probabilidades.ravel(), respuestas.ravel(), y.ravel()):
    print(f"{entrada.astype(int)} -> probabilidad {prob:.4f} -> respuesta: {respuesta} (esperado {int(esperado)})")

aciertos = int(np.sum(respuestas == y))
print(f"Aciertos: {aciertos}/10")

assert np.array_equal(respuestas, y.astype(int))

# ---------------------------------------------------------------
# Pruebas con condiciones nuevas
# ---------------------------------------------------------------
X_nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)

print("\nPruebas con condiciones nuevas:")
prob_nuevos = probabilidad(X_nuevos, pesos, sesgo)
decision_nuevos = (prob_nuevos >= 0.5).astype(int)

for entrada, prob, decision in zip(X_nuevos, prob_nuevos.ravel(), decision_nuevos.ravel()):
    texto = "regar" if decision == 1 else "no regar"
    print(f"Humedad {entrada[0]:.0f}% | Temperatura {entrada[1]:.0f}°C -> "
          f"probabilidad {prob:.4f} -> decision: {decision} ({texto})")

# ---------------------------------------------------------------
# Experimentos con los parámetros
# ---------------------------------------------------------------
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas epocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Mas epocas", 20000, 0.5),
    ("Tasa pequena", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

caso_45_34 = np.array([[45, 34]], dtype=float)

print("\nExperimentos con los parametros:")
print(f"{'Prueba':<20} | {'Epocas':>6} | {'Tasa':>5} | {'Error final':>11} | {'Aciertos':>8} | P(45%,34°C)")
for nombre, ep, tasa in experimentos:
    w, b, e = entrenar(tasa, ep)
    correctas = int(np.sum((probabilidad(X, w, b) >= 0.5).astype(int) == y))
    p = probabilidad(caso_45_34, w, b)[0, 0]
    print(f"{nombre:<20} | {ep:>6} | {tasa:>5} | {e:>11.6f} | {correctas:>5}/10 | {p:.4f}")

# ---------------------------------------------------------------
# Prueba adicional con el umbral (sin volver a entrenar)
# ---------------------------------------------------------------
print("\nComparacion de umbrales (mismos pesos):")
print(f"{'Entrada':<14} | {'Prob':>6} | u=0.4 | u=0.5 | u=0.6")
todos = np.vstack([X, X_nuevos])
for entrada, prob in zip(todos, probabilidad(todos, pesos, sesgo).ravel()):
    r = [int(prob >= u) for u in (0.4, 0.5, 0.6)]
    cambia = "  <- cambia" if len(set(r)) > 1 else ""
    print(f"{entrada[0]:>3.0f}% , {entrada[1]:>2.0f}°C   | {prob:.4f} | "
          f"{r[0]:>5} | {r[1]:>5} | {r[2]:>5}{cambia}")

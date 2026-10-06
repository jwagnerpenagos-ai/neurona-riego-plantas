# Neurona artificial para el riego de plantas

Taller de neurona artificial. La neurona recibe la humedad del suelo (%) y la temperatura (°C) y decide si hay que regar la planta (1) o no (0). Usa la función sigmoide y se entrena con descenso del gradiente, igual que el perceptrón de la compuerta AND que vimos en clase.

Los datos son de ejemplo, no son una recomendación real para ninguna planta.

## Cómo ejecutarlo

```
uv sync
uv run main.py
```

La salida completa está en `salida_consola.txt`.

## Datos

| Caso | Humedad | Temperatura | Salida |
|---|---|---|---|
| 1 | 80 % | 18 °C | 0 |
| 2 | 70 % | 22 °C | 0 |
| 3 | 65 % | 28 °C | 0 |
| 4 | 55 % | 25 °C | 0 |
| 5 | 50 % | 32 °C | 0 |
| 6 | 40 % | 30 °C | 1 |
| 7 | 35 % | 25 °C | 1 |
| 8 | 30 % | 32 °C | 1 |
| 9 | 20 % | 35 °C | 1 |
| 10 | 10 % | 38 °C | 1 |

Antes de entrenar se dividen las entradas por `escala = [100, 50]` para que queden entre 0 y 1. Los datos nuevos se dividen por la misma escala.

## Resultado del entrenamiento base

Con tasa de aprendizaje 0.5 y 10000 épocas:

- Peso de la humedad: -19.5032
- Peso de la temperatura: 2.3101
- Sesgo: 7.4022
- Error final: 0.019645
- Aciertos: 10 de 10

El peso de la humedad es negativo: entre más húmedo está el suelo, menos probable es que haya que regar. El de la temperatura es positivo: entre más calor, un poco más probable es regar. La humedad pesa mucho más que la temperatura (19.5 contra 2.3).

## Pruebas con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 (no regar) |
| 45 % | 34 °C | 0.5490 | 1 (regar) |
| 25 % | 22 °C | 0.9719 | 1 (regar) |
| 50 % | 25 °C | 0.2325 | 0 (no regar) |
| 30 % | 40 °C | 0.9677 | 1 (regar) |

El caso de 45 % y 34 °C es el más dudoso porque queda justo en el medio, apenas por encima de 0.5.

## Experimentos

| Prueba | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | Rápido |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | Insuficiente |
| Intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | Lento |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | Rápido |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | Insuficiente |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | Lento |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | Rápido |

Todas las pruebas acertaron los 10 casos, así que lo que cambia es el error. También vimos que 20000 épocas con tasa 0.5 dan prácticamente lo mismo que 10000 épocas con tasa 1.0.

## Prueba con el umbral

Con los mismos pesos, sin volver a entrenar, probamos umbrales de 0.4, 0.5 y 0.6. El único caso que cambió fue 45 % y 34 °C: con 0.4 y 0.5 da regar, pero con 0.6 da no regar, porque su probabilidad es 0.549. Los demás casos tienen probabilidades lejos de esos valores y no cambian.

Los pesos no cambian porque el umbral se usa después de calcular la probabilidad. Solo cambia desde qué valor decimos que hay que regar.

## Análisis

**1. ¿Por qué fue necesario normalizar la humedad y la temperatura?**
Porque la humedad y la temperatura están en escalas distintas (0 a 100 y 0 a 50). Si las dejamos así, z se vuelve muy grande, la sigmoide se satura cerca de 0 o 1 y su derivada p·(1 − p) queda casi en cero, entonces el aprendizaje se frena. Además el gradiente de cada peso depende del valor de su entrada, así que la humedad dominaría las actualizaciones solo por tener números más grandes. Al dividir por [100, 50] las dos quedan cerca de 0 a 1 y la misma tasa de aprendizaje sirve para los dos pesos. En la compuerta AND esto no hacía falta porque las entradas ya eran 0 y 1.

**2. ¿En qué operaciones se utilizó X_normalizado y para qué se conservó X?**
X_normalizado se usa en la suma ponderada (z = X_normalizado @ pesos + sesgo) y en el gradiente de los pesos (gradiente_pesos = X_normalizado.T @ gradiente_z). X lo conservé para mostrar los resultados en % y °C, que es como se entienden. Los datos nuevos también los dejo en su escala original y la función probabilidad los divide por la misma escala antes de pasarlos por la neurona.

**3. ¿Qué ocurrió al utilizar solamente 100 épocas?**
El error quedó en 0.166, unas 8 veces más que el 0.020 de la prueba base. Los pesos casi no alcanzaron a moverse desde sus valores iniciales, entonces las probabilidades quedaron cerca de 0.5. Acertó los 10 casos, pero con muy poca seguridad: con un cambio pequeño en los datos o en el umbral las respuestas podrían cambiar. El aprendizaje fue insuficiente.

**4. ¿Más épocas siempre produjeron una mejora importante?**
No. De 100 a 1000 épocas el error bajó de 0.166 a 0.067, y de 1000 a 10000 bajó a 0.020, que sí son mejoras grandes. Pero de 10000 a 20000 épocas (el doble de cálculo) solo bajó de 0.020 a 0.012 y los aciertos siguieron en 10/10. La mejora se va haciendo más pequeña porque, cuando las probabilidades se acercan a 0 y 1, la derivada de la sigmoide se hace pequeña y los gradientes también.

**5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?**
Con 0.01 aprendió muy lento: después de 10000 épocas el error seguía en 0.130, casi tan alto como con solo 100 épocas y tasa 0.5 (0.166). Cada paso mueve tan poco los pesos que harían falta muchísimas más épocas para llegar a lo de la prueba base. Con 0.1 el error llegó a 0.0497, mejor, pero todavía más del doble que con 0.5.

**6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?**
Con 1.0 y 2.0 aprendió más rápido y terminó con menos error (0.0117 y 0.0066) en las mismas 10000 épocas, sin volverse inestable y con 10/10 aciertos. Yo pensaba que con 2.0 iba a oscilar, pero no pasó: con los datos normalizados y el error cuadrático medio los gradientes de este problema son pequeños (la derivada de la sigmoide máximo da 0.25), entonces pasos de 1 o 2 no alcanzan a pasarse del mínimo. Con tasas mucho más grandes o en otros problemas sí puede pasar que los pesos salten de un lado a otro del mínimo y el error suba y baje sin estabilizarse.

**7. ¿Qué representa el signo del peso correspondiente a la humedad?**
Es negativo en todas las pruebas (−19.50 en la prueba base). Quiere decir que entre más humedad tenga el suelo, menor es la probabilidad de regar, que es lo esperado porque un suelo húmedo ya tiene agua. Como es tan grande comparado con el otro peso, la humedad es la variable que más pesa en la decisión.

**8. ¿Qué representa el signo del peso correspondiente a la temperatura?**
En la prueba base es positivo (+2.31): a mayor temperatura, mayor probabilidad de regar, lo que cuadra con que el calor aumenta la evaporación. Pero es pequeño comparado con el de la humedad (2.31 frente a −19.50). En los datos la temperatura alta casi siempre viene junto con humedad baja, y la humedad sola ya alcanza para separar los casos, así que la neurona no tiene suficientes ejemplos para aprender bien el efecto propio de la temperatura.

**9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?**
Porque la sigmoide da un valor continuo entre 0 y 1 (qué tan probable es que haya que regar), pero la acción es binaria: se riega o no se riega. Si esto controlara, por ejemplo, una válvula con un microcontrolador, necesitaría una orden concreta. El umbral convierte la probabilidad en esa orden, y además se puede mover según el riesgo que uno quiera asumir sin tener que volver a entrenar.

**10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?**

- **Pocos datos:** solo hay 10 ejemplos didácticos, no medidos en plantas reales.
- **Pocas variables:** no tiene en cuenta la especie, el tipo de suelo, la humedad del aire, el sol, el viento, la lluvia, la etapa de crecimiento ni la hora del día.
- **Frontera lineal:** una sola neurona solo puede trazar una línea recta entre regar y no regar, así que no representa relaciones más complejas.
- **Variables relacionadas:** como la temperatura alta casi siempre va con humedad baja, la neurona no logra separar bien el efecto de cada una.
- **No dice cuánto regar:** solo decide sí o no, no la cantidad de agua ni el tiempo.
- **Fuera de rango:** con temperaturas bajo 0 °C o por encima de 50 °C sus predicciones no son confiables.
- **Sin validación aparte:** se evaluó con los mismos datos del entrenamiento.
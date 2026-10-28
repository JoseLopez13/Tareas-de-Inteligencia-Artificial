# **Reporte Breve: Q-learning en Cliff Walking**

**1. ¿Por qué el camino de 13 pasos, pegado al acantilado, es óptimo?**
En este entorno cada paso penaliza con -1 y no hay descuento temporal ($\gamma = 1$). El objetivo es llegar a la meta en la menor cantidad de pasos posibles. El camino geométrico más corto esquivando el acantilado es subir una casilla, avanzar 11 al Este y bajar una, sumando exactamente 13 pasos (retorno de -13). Cualquier otra ruta más alejada del borde requiere dar pasos extra, lo que acumula más castigos y empeora el retorno.

**2. ¿Por qué la curva de entrenamiento queda por debajo de −13 aunque la política greedy ya sea óptima?**
Porque durante el entrenamiento usamos una política $\epsilon$-greedy con $\epsilon = 0.1$, obligando al agente a explorar con una acción aleatoria el 10% de las veces. Como la ruta óptima está pegada al acantilado, si esa acción aleatoria resulta ser "Sur", el agente cae y recibe un castigo masivo de -100. Estas caídas constantes bajan drásticamente el promedio de recompensa en la gráfica, a pesar de que la tabla Q ya conozca el camino ideal.

**3. En el paso que cae al acantilado, ¿cuál es s' en el backup de Q-learning, y por qué ese backup no trata el acantilado como estado terminal?**
Al caer, el estado siguiente ($s'$) evaluado en el backup es la casilla de inicio $S(3, 0)$. No se trata como un estado terminal porque las reglas del entorno dictan que el episodio no termina ahí; el agente simplemente es teletransportado al inicio. Por esto, el valor futuro esperado ($\max_{a'} Q(s', a')$) no es cero. El algoritmo evalúa el castigo inmediato de -100 sumado al costo estimado de tener que reiniciar todo el recorrido desde $S$.
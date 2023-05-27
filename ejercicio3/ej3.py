import numpy as np

# Función para obtener la recompensa de un estado
def obtener_recompensa(estado, laberinto):
    i, j = estado
    if laberinto[i][j] == -1:
        return -10
    elif laberinto[i][j] == 2:
        return 10
    else:
        return 0

# Función para obtener el estado siguiente dado un estado y una acción
def obtener_estado_siguiente(estado, accion, laberinto):
    # Acciones: 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
    i, j = estado
    if accion == 0 and i > 0:
        i -= 1
    elif accion == 1 and i < laberinto.shape[0] - 1:
        i += 1
    elif accion == 2 and j > 0:
        j -= 1
    elif accion == 3 and j < laberinto.shape[1] - 1:
        j += 1
    return (i, j) if laberinto[i][j] != -1 else estado

# Función para actualizar la tabla Q
def actualizar_Q(estado, accion, recompensa, estado_siguiente, laberinto):
    estado = estado[0]*laberinto.shape[1] + estado[1]
    estado_siguiente = estado_siguiente[0]*laberinto.shape[1] + estado_siguiente[1]
    Q[estado][accion] = Q[estado][accion] + alpha * (recompensa + gamma * np.max(Q[estado_siguiente]) - Q[estado][accion])

# Función para elegir la acción siguiente
def elegir_accion(estado, epsilon, laberinto):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_acciones) # Exploración: eligiendo una acción al azar
    else:
        return np.argmax(Q[estado[0]*laberinto.shape[1] + estado[1]]) # Explotación: eligiendo la acción con el valor Q más alto

# Función para obtener el camino óptimo
def obtener_camino_optimo(Q, laberinto):
    estado = (0, 0) # Comenzamos en la casilla de inicio
    camino_optimo = [estado]

    while laberinto[estado] != 2: # Mientras no lleguemos a la casilla de salida
        accion = np.argmax(Q[estado[0]*laberinto.shape[1] + estado[1]]) # Elegimos la acción con el valor Q más alto
        estado = obtener_estado_siguiente(estado, accion, laberinto) # Actualizamos el estado
        camino_optimo.append(estado) # Añadimos el estado al camino óptimo
        # print("Accion", accion, "Estado", estado)
        print("Accion", accion, "Estado", estado, "Camino optimo", camino_optimo)

    return camino_optimo


if __name__ == '__main__':
    # Representamos el laberinto como una matriz.
    # S = 1, E = 2, X = -1, y 0 representa una casilla vacía
    laberinto = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 2]
    ])
    # Número de estados y acciones
    num_estados = laberinto.shape[0] * laberinto.shape[1]
    num_acciones = 4

    # Tabla Q inicializada a ceros
    Q = np.zeros((num_estados, num_acciones))

    # Factor de descuento
    gamma = 0.9

    # Tasa de aprendizaje
    alpha = 0.1
    # Política epsilon-greedy
    epsilon = 1.0 # Inicializar epsilon a 1. Comenzamos con la exploración
    min_epsilon = 0.01 # Mínimo valor de epsilon
    decay_rate = 0.01 # Tasa de decaimiento de epsilon
    # Q-learning
    print("Entrando a q learning")
    for episodio in range(2000):
        # print("Episodio", episodio)
        estado = (0, 0)
        while laberinto[estado] != 2 and laberinto[estado] != -1:
            # print(Q[estado[0]*laberinto.shape[1] + estado[1]])
            accion = elegir_accion(estado, epsilon, laberinto) # Elegir la acción con el valor Q
            estado_siguiente = obtener_estado_siguiente(estado, accion, laberinto) # Obtener el estado siguiente
            recompensa = obtener_recompensa(estado_siguiente, laberinto) # Obtener la recompensa
            # print("Episodio",episodio,"Accion:",accion,"Estado actual (casilla)",estado,"Estado siguiente (casilla)", estado_siguiente,"Recompensa del estado siguiente", recompensa)
            actualizar_Q(estado, accion, recompensa, estado_siguiente, laberinto) # Actualizar la tabla Q
            estado = estado_siguiente # Actualizar el estado
        # Decaimiento de epsilon después de cada episodio
        epsilon = min_epsilon + (1 - min_epsilon)*np.exp(-decay_rate*episodio)
    print("Print tabla Q")
    print(Q)
    print("Comenzando a obtener camino")
    print("Acciones: 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha")
    # print(laberinto[0][0])
    camino_optimo = obtener_camino_optimo(Q, laberinto)
    
    print(camino_optimo)
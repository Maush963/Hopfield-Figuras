"""Red de Hopfield para reconocer figuras binarias sin librerías externas."""


def validar_matriz(matriz, filas, columnas):
    if len(matriz) != filas:
        raise ValueError("La cantidad de filas no coincide con M.")
    for fila in matriz:
        if len(fila) != columnas:
            raise ValueError("La cantidad de columnas no coincide con N.")
        for valor in fila:
            if valor not in (-1, 1):
                raise ValueError("Cada celda debe valer -1 o 1.")


def a_vector(matriz):
    vector = []
    for fila in matriz:
        for valor in fila:
            vector.append(valor)
    return vector


def a_matriz(vector, filas, columnas):
    matriz = []
    posicion = 0
    for _ in range(filas):
        fila = []
        for _ in range(columnas):
            fila.append(vector[posicion])
            posicion += 1
        matriz.append(fila)
    return matriz


def entrenar(patrones):
    """Calcula la matriz de pesos con la regla Hebb."""
    cantidad_neuronas = len(patrones[0])
    pesos = []
    for i in range(cantidad_neuronas):
        fila = []
        for j in range(cantidad_neuronas):
            suma = 0
            for patron in patrones:
                suma += patron[i] * patron[j]
            fila.append(0 if i == j else suma)
        pesos.append(fila)
    return pesos


def signo(valor, valor_anterior):
    # En un empate se conserva el estado anterior para evitar oscilaciones.
    if valor > 0:
        return 1
    if valor < 0:
        return -1
    return valor_anterior


def reconocer(entrada, pesos, max_iteraciones=20):
    """Actualiza todas las neuronas simultáneamente hasta estabilizarse."""
    estado = entrada[:]
    for _ in range(max_iteraciones):
        siguiente = []
        for i in range(len(estado)):
            activacion = 0
            for j in range(len(estado)):
                activacion += pesos[i][j] * estado[j]
            siguiente.append(signo(activacion, estado[i]))
        if siguiente == estado:
            return estado
        estado = siguiente
    return estado


def distancia(a, b):
    diferencias = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            diferencias += 1
    return diferencias


def figura_a_texto(matriz):
    lineas = []
    for fila in matriz:
        linea = ""
        for valor in fila:
            linea += "##" if valor == 1 else "  "
        lineas.append(linea)
    return "\n".join(lineas)


def leer_figura(filas, columnas):
    print("Introduce cada fila usando 1 para negro y 0 para blanco.")
    matriz = []
    for numero_fila in range(filas):
        while True:
            texto = input("Fila " + str(numero_fila + 1) + ": ").strip()
            valores = texto.split()
            if len(valores) != columnas:
                print("La fila debe contener " + str(columnas) + " valores.")
                continue
            if any(valor not in ("0", "1") for valor in valores):
                print("Solo se permiten 0 y 1.")
                continue
            matriz.append([1 if valor == "1" else -1 for valor in valores])
            break
    return matriz


def patrones_ejemplo():
    """Patrones 1, 2 y 4 del ejemplo de 8x5."""
    return [
        [
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ],
        [
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, -1, 1, -1, -1],
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
        ],
        [
            [1, 1, -1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ],
    ]


def main():
    patrones = patrones_ejemplo()
    filas = len(patrones[0])
    columnas = len(patrones[0][0])
    vectores = [a_vector(patron) for patron in patrones]
    pesos = entrenar(vectores)

    print("Red de Hopfield - reconocimiento de figuras")
    print("La matriz del ejemplo tiene " + str(filas) + " filas y " + str(columnas) + " columnas.")
    entrada = leer_figura(filas, columnas)
    vector_reconocido = reconocer(a_vector(entrada), pesos)
    reconocido = a_matriz(vector_reconocido, filas, columnas)

    distancias = []
    for numero, patron in enumerate(vectores, 1):
        distancias.append((distancia(vector_reconocido, patron), numero))
    distancias.sort()
    mejor_distancia, numero_patron = distancias[0]

    print("\nFigura reconstruida:")
    print(figura_a_texto(reconocido))
    print("\nPatrón reconocido: " + str(numero_patron))
    print("Celdas diferentes respecto al patrón: " + str(mejor_distancia))


if __name__ == "__main__":
    main()

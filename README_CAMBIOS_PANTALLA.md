# Cambios aplicados - Snake AI

## Nuevo flujo en pantalla del juego
Al iniciar, la ventana del juego ahora muestra:
- si existe un checkpoint previo
- juegos acumulados
- record
- promedio histórico
- opciones:
  - **T** = continuar entrenando
  - **M** = usar solo el modelo
  - **N** = empezar entrenamiento nuevo

## Modos
### TRAIN
- sigue entrenando
- guarda checkpoints periódicos
- guarda el mejor modelo cuando rompe récord
- retoma desde donde estaba si existe checkpoint

### MODEL
- usa el modelo cargado
- no entrena
- epsilon queda en 0
- útil para ver jugar al agente sin exploración aleatoria

## Persistencia agregada
Se guarda en `./model/last_checkpoint.pth`:
- pesos del modelo
- estado del optimizador
- número de juegos
- récord
- epsilon
- historial de scores
- historial de mean scores
- memoria de replay
- contador de estancamiento

El mejor modelo se guarda aparte en:
- `./model/model.pth`

## Estado visible dentro del juego
En la parte superior de la ventana ahora se muestra:
- Score
- Record
- Games
- Mean
- Epsilon
- Mode
- Loaded

## Epsilon dinámico
Ya no depende solo de `n_games`.
Ahora ajusta exploración según:
- cantidad de partidas
- rendimiento reciente
- estancamiento respecto al récord

## Correcciones incluidas además
- device CPU/GPU robusto
- carga con `map_location`
- corrección de colisiones proyectadas
- corrección de índice de acción en entrenamiento por lote
- reward por paso reiniciado correctamente
- helper de gráfico más robusto

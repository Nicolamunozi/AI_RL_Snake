# Snake AI - cambios aplicados

## Errores corregidos
- Se eliminó la dependencia forzada de CUDA. Ahora el proyecto corre en GPU si está disponible y si no, usa CPU.
- Se corrigió el cálculo de colisiones futuras en `get_state()`.
- Se corrigió `is_collision()`: antes ignoraba las coordenadas recibidas al revisar colisión con la cola.
- Se corrigió el entrenamiento por lotes en `QTrainer`: antes actualizaba mal el índice de acción.
- Se corrigió la carga del modelo con `map_location`, evitando errores entre CPU y GPU.
- Se corrigió la recompensa por paso: antes podía arrastrar el reward anterior.
- Se eliminó el uso de una lista mutable como parámetro por defecto en `play_game()`.
- Se hizo más robusto el reinicio del juego, evitando reconstrucciones innecesarias de pantalla.
- Se evita que la comida reaparezca encima de la serpiente.

## Mejora funcional
- Se agregó una penalización pequeña por paso (`-0.1`) para incentivar rutas más eficientes.
- Se mejoró la compatibilidad del gráfico fuera de notebooks.

## Archivos principales
- `main.py`: agente, estado, política y loop de entrenamiento.
- `model.py`: red neuronal y entrenamiento Q-learning.
- `snake.py`: lógica del juego.
- `food.py`: generación de comida.
- `scoreboard.py`: puntaje.
- `helper.py`: gráfico de entrenamiento.

explicación teórica con dibujos: A System One model evaluates a state and returns typed answers and probabilities.

types of questions: choice (clasificación), score (regresión), noul (clasificación binaria)

explicar la diferencia con un clasificador/regresor fijo

El nombre hace referencia al economista William Stanley Jevons — la idea es que a medida que el costo de una decisión baja, la demanda de decisiones explota.

las responde todas en un único paso paralelo, no es secuencial
unlike traditional LLMs, Jev is neither constrained by text generation or sequential decision making!

sin embargo, jev no sustituye a los llms, por ejemplo para un chatbot asistente o un agente de código

cuán faster y cheaper es

ejemplo
answer
{
  "model": "jev-latest",
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency or time-sensitivity"
    }
  }
}
response
{
  "is_urgent": {
    "type": "noul",
    "noul": 0.999
  }
}

reinforcement learning for calibrated decisions (RLCD)
optimiza sus probabilidades frente a resultados en lugar de preferencias humanas (LLMs, RLHF).
RLHF optimizes for sounding right to a person, while RLCD optimizes for being right
llms tend to be overconfident, jev is calibrated: higher confidence means higher accuracy. a decision tagged 90% confidence should actually be correct around 90% of the time
El benchmark también usa el promedio de GPT-6 Astra y Fable 5.1 como referencia "correcta", lo que introduce un sesgo hacia esos modelos. mirar mejor esto. lo usan como etiquetas de entrenamiento o de test?

https://typesafe.ai/blog/introducing-system-one-models-and-jev
primer gif vs llm normal
accuracy vs cost figure
tool call error figure: The numbers for LLMs are from OpenRouter i.e., there almost certainly is bias here: more complex queries might be routed to better models.

casos de uso: tool calling, structured outputs, llm evals, el juego de shooter y wiki racing...

decir que para ciertas tareas puede ser mejor entrenar un clasificador ad-hoc. jev da más flexibilidad y es más general

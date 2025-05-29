## Reto: **"Despliegue de modelo predictivo de Fallos en Equipos en Plantas Energéticas a 15 días vista"**

**Objetivo del Reto:**  
Evolucionar la solución para desarrollar un pipeline de entrega del modelo analítico predictivo que permita anticipar fallos a 15 días vista en los equipos de la planta energética. 
- El pipeline debe estar compuesto por componentes reutilizables.
- Se debe disparar una vez por semana, todos los lunes a las 08:00 horas.
- Debe incluir el preprocesado, entrenamiento, evaluación, registro y despliegue del modelo.
- Asimismo se debe evaluar el modelo desplegado.
- En las distintas evaluaciones, el modelo debería cumplir con el requerimiento definido ( una precisión mínima del **80% (accuracy ≥ 0.80)** y por lo menos un **0.75 de f1**. ), si no se debe enviar una alerta.


Para lograr esto, deberéis:

1. **Generar los assets** necesarios en tu workspace.  
2. **Identificar y construir** de los componentes del pipeline.  
3. **Definir el pipeline** para usar los componetne.
5. **Documentar el proceso**.  


#### **NOTAS:** 
- Usar el kernel: **Python 3.8 - AzureML (Python 3.10.11)**
- Usar los compute assets provistos


### **Estructura de carpetas:**
- Carpeta **.** para los `jupyter notebooks`  
- Carpeta **src** para los `scripts y yaml` de trabajo  
- Ejemplo: 

```
.
└── repositorio/
    ├── [archivos jupyter]
    └── src/
        └── [yaml y scripts python]
```


### **Trabajo en equipo:**  
   - Usar un repositorio el común.
      - Trabajar en la rama "etapa 4"
   - Trabajo por pares:
      - Recomendable abordar las tareas por pares.
   - Dividir las tareas.
   - Recomendaciones:
      - Evitar trabajar en el mismo archivo a la vez. Especialmente los jupyter notebooks.
      - Comunicarse cuando haya commits y pushes para realizar pulls. Evitar en lo posible conflictos.

### **Entregables:**
- Repositorio que contenga:
1. **Jupyter Notebooks conteniendo:**
   - Código documentado y estructurado.  
   
2. **Reflexión:** sobre los desafíos encontrados y pasos futuros.

### **Criterios de Evaluación:**  

- **Claridad en el código**: Los archivos debe reflejar un proceso estructurado.
- **Resultado final**: Los componentes generados en el workspace.
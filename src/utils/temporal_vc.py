import numpy as np
from sklearn.model_selection import BaseCrossValidator

class TemporalSplit(BaseCrossValidator):
    """
    Validación cruzada temporal para datos ordenados cronológicamente.
    
    Parámetros:
    - test_size: Tamaño del conjunto de prueba para cada iteración.
    - n_splits: Número de particiones (iteraciones) para la validación cruzada.
    """
    def __init__(self, test_size, n_splits):
        self.test_size = test_size
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """
        Genera índices para dividir los datos en entrenamiento y prueba.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calcula el tamaño del conjunto de entrenamiento para cada iteración
        train_size = n_samples - self.test_size
        
        for i in range(self.n_splits):
            # Define los índices de entrenamiento y prueba
            train_idx = indices[:train_size]
            test_idx = indices[train_size:train_size + self.test_size]
            
            yield train_idx, test_idx
            
            # Mueve la ventana hacia adelante
            train_size += self.test_size
            if train_size >= n_samples:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Devuelve el número de particiones (iteraciones).
        """
        return self.n_splits
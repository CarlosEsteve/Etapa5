import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate
import pandas as pd

# Train de CV
def cv_train(name, pipeline, cv, X_train, y_train, X_test, y_test):
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="accuracy", return_estimator=True, return_train_score=True, n_jobs=5)
    trained_model = cv_results["estimator"][0]
    scores = pd.DataFrame(cv_results)

    print("test score (mean-std): {0:.2f} - {1:.2f}".format(scores["test_score"].mean(), scores["test_score"].std()))
    print("train score (mean-std): {0:.2f} - {1:.2f}".format(scores["train_score"].mean(), scores["train_score"].std()))
    print("params:", pipeline.named_steps.get("classifier").get_params())

    y_pred = trained_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return {"acc": round(scores["test_score"].mean(), 2), "f1": round(f1, 2),}


# Curvas de aprendizaje
def generate_learning_curves(name, pipeline, X, y, train_sizes, cvss):
    results = learning_curve(pipeline, X, y, train_sizes=train_sizes,
                             cv=cvss, scoring='accuracy', n_jobs=5)
    
    train_size, train_scores, test_scores = results[:3]

    # graficar la curva.
    plt.errorbar(train_size, train_scores.mean(axis=1),
                 yerr=train_scores.std(axis=1), label="Error de entrenamiento")
    plt.errorbar(train_size, test_scores.mean(axis=1),
                 yerr=test_scores.std(axis=1), label="Error de prueba")
    plt.legend()

    plt.xscale("log")
    plt.xlabel("Número de muestras en el conjunto de entrenamiento")
    plt.ylabel("Accuracy")
    plt.title("Curva de aprendizaje para {name}".format(name=name))

    plt.show()


# Curvas de validación
def generate_validation_curves(name, pipeline, X, y, param_name, param_range, cvss):
    train_scores, test_scores = validation_curve(
        pipeline, X, y, param_name=param_name, param_range=param_range,
        cv=cvss, scoring="accuracy", n_jobs=5)

    # graficar la curva.
    plt.plot(param_range, train_scores.mean(
        axis=1), label="Error de entrenamiento")
    plt.plot(param_range, test_scores.mean(axis=1), label="Error de prueba")
    plt.legend()

    plt.xlabel("Valor del ({param_name})".format(
        param_name=param_name))
    plt.ylabel("Accuracy")
    plt.title("Curva de validación para {name}".format(name=name))

    plt.show()
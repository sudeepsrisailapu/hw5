# add your code to this file
import sys
import pandas as pd
import matplotlib.pyplot as pyplt
import numpy as np


def data_plot(file_name):
    data = pd.read_csv(file_name)

    years = data['year']
    days = data['days']

    pyplt.plot(years, days)

    pyplt.xlabel('Year')
    pyplt.ylabel('Number of Frozen Days')

    pyplt.savefig("data_plot.jpg")
    pyplt.close()


def gradient_descent(X_normalized, y, learning_rate, iterations):
    m = len(y)
    weights = np.zeros(X_normalized.shape[1])
    history = []
    weight_history = []

    weight_history.append(weights.copy())

    for i in range(iterations):
        prediction = X_normalized.dot(weights)
        errors = prediction - y

        gradient = (1 / m) * X_normalized.T.dot(errors)
        weights -= learning_rate * gradient

        cost = (1/(2*m)) * np.sum(errors**2)
        history.append(cost)

        if i % 10 == 0 and i != 0:
            weight_history.append(weights.copy())

    print("Q5a:")
    for w in weight_history:
        print(w)

    print(f"Q5b: .27")
    print(f"Q5c: 500")


    pyplt.plot(range(iterations), history)
    pyplt.xlabel("Iterations")
    pyplt.ylabel("MSE Loss")
    pyplt.savefig("loss_plot.jpg")

    return weights, history


def data_normalized(filename):
    file_data = pd.read_csv(filename)

    years = file_data['year'].values

    m = np.min(years)
    M = np.max(years)

    years_normalized = (years - m) / (M - m)

    X_normalized = np.c_[years_normalized, np.ones(len(years))]

    print("Q3:")
    print(X_normalized)

    return X_normalized, m, M


def closed_form_solution(X_normalized, y):
    X_inverse = np.linalg.inv(X_normalized.T.dot(X_normalized))
    transpose_product = X_normalized.T.dot(y)

    weights = X_inverse.dot(transpose_product)

    print("Q4:")
    print(weights)

    return weights


def prediction(weights, m, M):
    normalize_year = (2023-m) / (M - m)
    augment_normalization = np.array([normalize_year, 1])

    y_prediction = augment_normalization.dot(weights)

    print("Q6: " + str(y_prediction))

    return y_prediction


def interpretation(weights):
    w = weights[0]
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="

    print(f"Q7a: {symbol}")
    print("Q7b: If w > 0, frozen days increase with time. If w < 0, frozen days decrease with time. "
          "If w = 0, no relationship is present")
    return


def prediction_no_freeze(weights, m, M):
    w, b = weights

    normalized_x_star = -b / w
    x_star = normalized_x_star * (M - m) + m

    print(f"Q8a: {x_star}")
    print("Q8b: The value received from the prediction is 2463,"
          " meaning that in the year 2463 is approximately when the lake will no longer freeze over. "
          "This means that there is around 400 years where the lake will freeze.")

    return x_star


def main():
    file_name = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    data_plot(file_name)

    X_normalized, m, M = data_normalized(file_name)

    data = pd.read_csv(file_name)
    y = data['days'].values

    closed_weights = closed_form_solution(X_normalized, y)

    gd_weights, history = gradient_descent(X_normalized, y, learning_rate, iterations)

    first_prediction = prediction(closed_weights, m, M)

    interpretation(closed_weights)

    second_prediction = prediction_no_freeze(closed_weights, m, M)


if __name__ == "__main__":
    main()








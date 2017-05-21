import numpy as np

from collections import deque
from scipy.optimize import curve_fit

class Line():
    def __init__(self, coeffs=None, n_history=40):
        # was the line detected in the last iteration?
        self.detected = False
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.coeffs = coeffs if coeffs is not None else np.array([])
        # history of last n_history quadratic functions
        self.history = deque([], n_history)
        #difference in fit coefficients between last and new fits
        self.coeff_diffs = np.array([0, 0, 0], np.float)

    @property
    def current_fit(self):
        return self.coeffs

    @property
    def last_fit(self):
        best = list(filter(lambda h: h[0], self.history))
        return best[-1][1] if len(best) > 0 else None

    @property
    def average_fit(self):
        best = list(filter(lambda h: h[0], self.history))
        if len(best) < 2:
            return self.last_fit
        upper_limit = 10
        max = upper_limit if len(best) >= upper_limit else len(best)
        weights = np.array([1 for i in range(max, 0, -1)])
        #weights = 1 / weights
        coeffs = np.array([h[1] for h in best[-max:]])
        avg = np.average(coeffs, weights=weights, axis=0)
        return avg

    @staticmethod
    def create_function(a, b, c):
        def _quadratic(x):
            return quadratic2(x, a, b, c)
        return _quadratic

    @property
    def function(self):
        return self.create_function(*self.coeffs)

    def fit(self, centroids):
        if len(centroids) < 3:
            return None

        xs = [p[0] for p in centroids]
        ys = [p[1] for p in centroids]
        coeffs = self._fit_function(xs, ys)
        #coeffs = _transform_quadratic(*coeffs) if coeffs is not None else None
        return np.array(coeffs)

    def accept_fit(self, coeffs):
        self.detected = True
        self.coeff_diffs = coeffs - self.coeffs if len(self.coeffs) == 3 else coeffs
        self.coeffs = coeffs.copy()
        self.history.append((True, coeffs))

    def reject_fit(self, coeffs):
        self.detected = False
        if coeffs is not None:
            self.coeff_diffs = coeffs - self.coeffs if len(self.coeffs) == 3 else coeffs
            self.coeffs = coeffs.copy()
        self.history.append((False, coeffs))

    def _find_last_detected(self):
        detected_lines = list(filter(lambda f: f[0], self.history))
        return detected_lines[-1][1] if len(detected_lines) > 0 else None


    @staticmethod
    def _fit_function(xs, ys):
        return np.polyfit(ys, xs, 2)

    @staticmethod
    def _fit_circle(xs, ys):
        return curve_fit(circle, ys, xs)


def circle(x, xc, yc, r):
    return np.sqrt(r*r - (x-xc)*(x-xc)) + yc

def quadratic(x, a, b, c):
    return a*(x + b)**2 + c

def quadratic2(x, a, b, c):
    return a*x**2 + b*x + c

def _transform_quadratic(a, b, c):
    '''transforms the polynomial a*x^2 + b*x + c into the form a * (x + b)^2 + c'''
    return (a, b/(2*a), (c - b**2/(4*a)))
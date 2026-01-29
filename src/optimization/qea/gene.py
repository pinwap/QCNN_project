import numpy as np


class QuantumGene:
    """
    Represents a single Quantum Gene in a QEA.
    A gene is represented as a qubit (alpha|0> + beta|1>).
    """

    def __init__(self) -> None:
        # Start at pi/4 (Uniform Superposition)
        self.theta: float = np.pi / 4
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.update_amplitudes()

    def update_amplitudes(self) -> None:
        # Calculate alpha, beta amplitudes from theta
        self.alpha = float(np.cos(self.theta))
        self.beta = float(np.sin(self.theta))

    def observe(self) -> int:
        # Measure result from chromosome alpha|0> + beta|1>
        prob_1 = self.beta**2
        return 1 if np.random.rand() < prob_1 else 0

    def rotate(
        self,
        current_bit: int,
        best_bit: int,
        fitness_current: float,
        fitness_best: float,
    ) -> None:
        """
        Rotate the state towards the target based on the QEA lookup table.
        """
        is_current_F_better = fitness_current >= fitness_best

        d_theta = 0.0
        sign = 0
        delta = 0.03 * np.pi
        eps = 1e-6

        # QEA Rotation Table
        if current_bit == 0 and best_bit == 0:
            d_theta = 0.0
        elif current_bit == 0 and best_bit == 1:
            if not is_current_F_better:
                d_theta = 0.0
            else:
                d_theta = delta
                if abs(self.beta) < eps:
                    sign = 0
                elif abs(self.alpha) < eps:
                    sign = int(np.random.choice([-1, 1]))
                elif self.alpha * self.beta > 0:
                    sign = -1
                elif self.alpha * self.beta < 0:
                    sign = 1
        elif current_bit == 1 and best_bit == 0:
            if not is_current_F_better:
                d_theta = delta
                if abs(self.beta) < eps:
                    sign = 0
                elif abs(self.alpha) < eps:
                    sign = int(np.random.choice([-1, 1]))
                elif self.alpha * self.beta > 0:
                    sign = -1
                elif self.alpha * self.beta < 0:
                    sign = 1
            else:
                d_theta = delta
                if abs(self.alpha) < eps:
                    sign = 0
                elif abs(self.beta) < eps:
                    sign = int(np.random.choice([-1, 1]))
                elif self.alpha * self.beta > 0:
                    sign = 1
                elif self.alpha * self.beta < 0:
                    sign = -1
        elif current_bit == 1 and best_bit == 1:
            d_theta = delta
            if abs(self.alpha) < eps:
                sign = 0
            elif abs(self.beta) < eps:
                sign = int(np.random.choice([-1, 1]))
            elif self.alpha * self.beta > 0:
                sign = 1
            elif self.alpha * self.beta < 0:
                sign = -1

        # Apply rotation
        self.theta += sign * d_theta
        self.theta = self.theta % (2 * np.pi)
        self.update_amplitudes()

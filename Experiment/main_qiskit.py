import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment_template import ExperimentConfig, ExperimentRunner

logger = logging.getLogger(__name__)


def main():
    cfg = ExperimentConfig(
        backend="qiskit-qcnn",
        dataset="mnist",
        data_path="../data",
        n_train=400,
        n_test=100,
        preprocessors=["bilinear_resize_4x4", "flatten"],
        n_qubits=16,
        max_iter=200,
        save_outputs=False,
        script_name="main_qiskit",
    )

    runner = ExperimentRunner(cfg)
    result = runner.run()
    logger.info(result.summary)


if __name__ == "__main__":
    main()
    
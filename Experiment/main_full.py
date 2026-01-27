import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment_template import ExperimentConfig, ExperimentRunner

logger = logging.getLogger(__name__)


def main():
    cfg = ExperimentConfig(
        backend="qea-qcnn",
        dataset="mnist",
        data_path="../data",
        n_train=400,
        n_test=100,
        preprocessors=["bilinear_resize_4x4", "flatten"],
        n_qubits=16,
        n_pop=50,
        n_gen=20,
        n_gates=180,
        epochs=5,
        lr=0.01,
        save_outputs=True,
        script_name="main_full",
    )

    runner = ExperimentRunner(cfg)
    result = runner.run()

    logger.info("\n🏁 Experiment Finished!")
    logger.info(result.summary)


if __name__ == "__main__":
    main()

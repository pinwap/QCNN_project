
import sys
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Experiment.experiment_template import ExperimentConfig, ExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Benchmark")

def run_benchmark():
    # Define candidates
    preprocess_candidates = [
        ("Bilinear", ["bilinear_resize_4x4", "flatten"]),
        ("PCA", ["pca_16"]), 
        ("DCT", ["dct_keep_16"]) 
    ]
    encoding_candidates = ["angle", "pge", "zz"]
    
    results = []

    logger.info("Starting Benchmark: 3 Preprocessors x 3 Encodings = 9 Runs")

    for prep_name, prep_steps in preprocess_candidates:
        for enc_name in encoding_candidates:
            logger.info(f"Running: Preprocessor={prep_name}, Encoding={enc_name}")
            
            # Configure Experiment
            # Note: We use QEA to find structure, then RETRAIN with Qiskit for final score
            cfg = ExperimentConfig(
                backend="qea-qcnn",
                dataset="mnist",
                data_path="../data", # Adjust relative path
                n_train=200, # Reduced size for faster benchmarking
                n_test=50,
                preprocessors=prep_steps,
                n_qubits=16,
                encoding=enc_name,
                
                # QEA Settings (Fast search for benchmark)
                n_pop=20,
                n_gen=10, 
                n_gates=200,
                epochs=3, # Low epochs during search
                lr=0.01,
                
                # Retrain Settings (Full training on best structure)
                retrain_with_qiskit=True,
                max_iter=40, # 40 epochs for final retraining
                
                save_outputs=True,
                script_name=f"bench_{prep_name}_{enc_name}"
            )
            
            try:
                runner = ExperimentRunner(cfg)
                res = runner.run()
                
                # Extract results
                qea_acc = res.summary.get("best_accuracy", 0.0)
                
                retrain_res = res.summary.get("retrain_result", {})
                if retrain_res:
                    final_train_acc = retrain_res.get("train_score", 0.0)
                    final_test_acc = retrain_res.get("test_score", 0.0)
                else:
                    final_train_acc = 0.0
                    final_test_acc = 0.0
                
                logger.info(f"✅ Finished {prep_name}-{enc_name}: Test Acc = {final_test_acc:.4f}")
                
                results.append({
                    "Preprocessor": prep_name,
                    "Encoding": enc_name,
                    "QEA_Best_Acc": qea_acc,
                    "Final_Train_Acc": final_train_acc,
                    "Final_Test_Acc": final_test_acc
                })
                
            except Exception as e:
                logger.error(f"Failed {prep_name}-{enc_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "Preprocessor": prep_name,
                    "Encoding": enc_name,
                    "QEA_Best_Acc": 0.0,
                    "Final_Train_Acc": 0.0,
                    "Final_Test_Acc": 0.0
                })

    # Save Results
    df = pd.DataFrame(results)
    output_csv = "benchmark_results.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Benchmark results saved to {output_csv}")
    
    # Plotting
    try:
        plot_benchmark_results(df)
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

def plot_benchmark_results(df):
    sns.set_theme(style="whitegrid")
    
    # Plot Final Test Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x="Preprocessor", 
        y="Final_Test_Acc", 
        hue="Encoding",
        palette="viridis"
    )
    plt.title("Benchmarking QCNN: Preprocessor vs Encoding (Test Accuracy)")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("benchmark_test_accuracy.png")
    logger.info("Saved plot to benchmark_test_accuracy.png")
    # plt.show() # Skip plotting window to avoid blocking if headless

if __name__ == "__main__":
    run_benchmark()

import os
import sys

import torch

# Ensure src is in the path
sys.path.append(os.path.join(os.getcwd(), "src"))

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load .env file at the very beginning
load_dotenv()

from data import DataManager
from models.qcnn import StandardQCNN  # noqa: E402
from training.engines import HybridEngine, QiskitEngine  # noqa: E402
from training.evolution import EvolutionarySearch  # noqa: E402
from training.pipeline import ProductionPipeline  # noqa: E402
from training.strategies import HybridStrategy # QiskitStrategy  # noqa: E402
from utils import (  # noqa: E402
    initialize_output_dir,
    notify_job_status,
    plot_fitness_history,
    plot_training_metrics,
    save_experiment_data,
    save_model,
)

logger = logging.getLogger(__name__)

"""
Main entry point for executing training and evolutionary search tasks.
ตัวเชื่อมทุกอย่างเข้าด้วยกัน ใช้ hydra ในการจัดการ config file
"""

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 0. Initialize Old-School Logging Directory
    # cfg.task is now a DictConfig (loaded from yaml), so specific name is in cfg.task_name
    task_name_str = cfg.get("task_name", "unknown_task")

    # Update notification title dynamically
    if cfg.get("notifications") and cfg.get("digits"):
        cfg.notifications.title = f"binary class experiment {cfg.digits}"

    # Extract metadata for folder naming
    fm_name = cfg.feature_map_type
    # Take the first preprocessor or 'raw' if list is empty
    prep_list = list(cfg.preprocessors)
    prep_name = prep_list[0] if prep_list else "raw"

    # Check if resuming from an existing directory
    if cfg.get("resume_dir"):
        save_dir = cfg.resume_dir
        # Assumes the folder name IS the file_id (which is true based on initialize_output_dir logic)
        file_id = os.path.basename(os.path.normpath(save_dir)) 
        #normalizepath = จัดการ path ให้ถูกต้องตาม os นั้นๆ เปลี่ยน \ -> / ใน windows ตัด /ตัวท้ายออก
        # basename = ดึงชื่อไฟล์หลัง / สุดท้ายออกมา
        logger.info(f"Resuming experiment from existing directory: {save_dir}")
        logger.info(f"inferred file_id: {file_id}")
    else:
        save_dir, file_id = initialize_output_dir(
            task_name_str,
            base_output_dir=cfg.save_dir,
            preprocessor_name=prep_name,
            feature_map_name=fm_name,
            override_output_dir=cfg.get("output_dir"),
        )

    # 0.5 Filter configuration for privacy/clarity

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(config_dict, dict):
        if task_name_str == "evolution":
            config_dict.pop("engine", None)
        elif task_name_str == "train":
            config_dict.pop("evolution", None)
            config_dict.pop("strategy", None)

    logger.info("Initializing experiment with configuration:\n%s", OmegaConf.to_yaml(config_dict))

    # Create checkpoints directory
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Initialize Data
    data_mgr = DataManager(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        n_val=cfg.get("n_val", 0),
        preprocessors=list(cfg.preprocessors),
        target_dim=cfg.target_dim,
        target_labels=tuple(cfg.digits) if cfg.digits else None,
        binary_groups=cfg.get("binary_groups"),
    )

    data_tuple = data_mgr.get_data()
    x_val, y_val = None, None
    if len(data_tuple) == 6:
        x_train, x_val, x_test, y_train, y_val, y_test = data_tuple
    else:
        x_train, x_test, y_train, y_test = data_tuple

    # 2. Perform Task
    try:
        if task_name_str == "auto_evolution":
            logger.info("Starting Task: Auto-Evolution (Search + Retrain)")

            # Phase 1: Setup Evolution Search
            if cfg.get("strategy_type") == "hybrid_end_to_end":
                 # Not supported in auto_evolution yet, usually train-only
                 raise NotImplementedError("Auto evolution not supported for hybrid_end_to_end yet")
            elif cfg.strategy_type == "hybrid":
                strategy = HybridStrategy(
                    num_qubits=cfg.num_qubits,
                    epochs=cfg.eval_epochs,
                    lr=cfg.eval_lr,
                    device=cfg.get("device"),
                    gradient_method=cfg.get("gradient_method", "param_shift"),
                    feature_map=cfg.feature_map_type,
                )
            else:
                strategy = QiskitStrategy(
                    num_qubits=cfg.num_qubits,
                    max_iter=cfg.eval_max_iter,
                    feature_map=cfg.feature_map_type,
                )

            search = EvolutionarySearch(
                data_manager=data_mgr,
                strategy=strategy,
                n_pop=cfg.n_pop,
                n_gen=cfg.n_gen,
                n_gates=cfg.n_gates,
            )

            # Phase 2: Setup Production Engine (Retrainer)
            production_engine = HybridEngine(
                feature_map=cfg.feature_map_type,
                epochs=cfg.train_epochs,
                lr=cfg.train_lr,
                gradient_method=cfg.get("gradient_method", "param_shift"),
                use_v2_primitives=cfg.get("use_v2_primitives", False),
                use_scheduler=cfg.get("use_scheduler", True),
                scheduler_patience=cfg.get("scheduler_patience", 5),
                scheduler_factor=cfg.get("scheduler_factor", 0.5),
                device=cfg.get("device"),
            )

            # Execution: Run Auto-Evolution Workflow
            from training import AutoEvolutionPipeline

            auto_pipeline = AutoEvolutionPipeline(
                evolution_search=search,
                production_engine=production_engine,
                num_qubits=cfg.num_qubits,
            )

            results = auto_pipeline.run(
                x_train, y_train, x_test, y_test, x_val, y_val,
                checkpoint_dir=checkpoints_dir, file_id=f"{file_id}_phase2"
            )
            final_score, history_list, trained_model, best_chromo, evolution_history = results

            # Save Combined Results
            # Phase 1: Evolution
            if best_chromo:
                save_experiment_data(
                    final_score=best_chromo.fitness,
                    history=evolution_history,
                    save_dir=save_dir,
                    file_id=f"{file_id}_phase1_evolution",
                    best_structure_code=best_chromo.collapse(),
                    config=config_dict,
                )
                plot_fitness_history(
                    history=evolution_history,
                    save_path=os.path.join(save_dir, "plots", f"{file_id}_phase1_fitness.png"),
                    title=f"Evolution Phase Best Fitness: {best_chromo.fitness:.4f}",
                )
                # Save the actual chromosome object for future loading
                save_model(best_chromo, save_dir, file_id, name="best_chromosome")

            # Phase 2: Retraining
            save_experiment_data(
                final_score=final_score,
                history=history_list,
                save_dir=save_dir,
                file_id=f"{file_id}_phase2_retrain",
                config=config_dict,
            )
            dataset_info = f"{cfg.n_train} {cfg.get('n_val', 0)}"
            plot_title = f"AutoEvolution - {prep_name} - {fm_name} - {dataset_info}"
            plot_training_metrics(
                history=history_list,
                save_dir=os.path.join(save_dir, "plots"),
                file_id=f"{file_id}_phase2",
                final_score=final_score,
                title=plot_title,
            )
            save_model(trained_model, save_dir, file_id, name="retrained_qcnn")

        elif task_name_str == "train":
            logger.info("Starting Task: Standard Training")

            if cfg.get("strategy") == "hybrid_end_to_end":
                
                from training.strategies.hybrid_end_to_end import HybridEndToEndStrategy
                logger.info("Using Hybrid End-to-End Strategy")
                
                strategy = HybridEndToEndStrategy(
                    num_qubits=cfg.n_qubits, # Use n_qubits from root config usually, or cfg.num_qubits if standard
                    epochs=cfg.train_epochs,
                    lr=cfg.train_lr,
                    encoding_dim=cfg.get("encoding_dim", cfg.target_dim),
                    device=cfg.get("device"),
                    pretrained_encoder_path=cfg.get("pretrained_encoder_path")
                )
                
                # Run evaluation directly as training
                final_score, history_list, model_state = strategy.evaluate(
                    structure_code=[],
                    x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    x_val=x_val, y_val=y_val
                )
                
                # Save
                torch.save(model_state, os.path.join(save_dir, "model", f"{file_id}_hybrid_e2e.pth"))
            
            else:
                # Resolve Engine
                if cfg.get("engine_type", "hybrid") == "hybrid":
                    engine = HybridEngine(
                        feature_map=cfg.feature_map_type,
                        epochs=cfg.train_epochs,
                        lr=cfg.train_lr,
                        gradient_method=cfg.get("gradient_method", "param_shift"),
                        use_v2_primitives=cfg.get("use_v2_primitives", False),
                        use_scheduler=cfg.get("use_scheduler", True),
                        scheduler_patience=cfg.get("scheduler_patience", 5),
                        scheduler_factor=cfg.get("scheduler_factor", 0.5),
                        device=cfg.get("device"),
                    )
                else:
                    engine = QiskitEngine(
                        feature_map=cfg.feature_map_type,
                        max_iter=cfg.get("train_max_iter"),  # Map max_iter for Qiskit
                    )

                # Resolve Model
                if cfg.model_type == "standard":
                    model = StandardQCNN(num_qubits=cfg.num_qubits)
                elif cfg.model_type == "evolutionary":
                    # Check for chromosome_path in config
                    chromo_path = cfg.get("chromosome_path")
                    if chromo_path and os.path.exists(chromo_path):
                        # Load the chromosome (it was saved with torch.save)
                        # We only need the collapsed genes (list of ints)
                        from models.qcnn import EvolutionaryQCNN
                        chromo_obj = torch.load(chromo_path, weights_only=False)
                        # If it's a QuantumChromosome object, collapse it;
                        # otherwise assume it's already a list
                        if hasattr(chromo_obj, "collapse"):
                            struct_code = chromo_obj.collapse()
                        elif hasattr(chromo_obj, "genes"):
                            struct_code = chromo_obj.genes
                        else:
                            struct_code = chromo_obj

                        model = EvolutionaryQCNN(num_qubits=cfg.num_qubits, chromosome=struct_code)
                        logger.info(f"Loaded evolutionary model from {chromo_path}")
                    else:
                        raise ValueError(
                            "model_type='evolutionary' requires a valid 'task.chromosome_path'"
                        )
                else:
                    raise ValueError(f"Unknown model_type: {cfg.get('model_type', 'standard')}")

                pipeline = ProductionPipeline(model=model, engine=engine)
                final_score, history_list, trained_model = pipeline.run(
                    x_train, y_train, x_test, y_test, x_val, y_val,
                    checkpoint_dir=checkpoints_dir, file_id=file_id
                )

            # Save Results

            save_experiment_data(
                final_score=final_score,
                history=history_list,
                save_dir=save_dir,
                file_id=file_id,
                config=config_dict,
            )
            dataset_info = f"{cfg.n_train} {cfg.get('n_val', 0)}"
            plot_title = f"{cfg.model_type} - {prep_name} - {fm_name} - {dataset_info}"
            plot_training_metrics(
                history=history_list,
                save_dir=os.path.join(save_dir, "plots"),
                file_id=file_id,
                final_score=final_score,
                title=plot_title,
            )
            save_model(trained_model, save_dir, file_id, name="trained_qcnn")

        else:
            logger.error("Unknown task: %s", task_name_str)
            return

        # Success Notification
        notify_job_status(
            status="success",
            task_name=task_name_str,
            fm_name=fm_name,
            prep_name=prep_name,
            score_or_error=final_score,
            file_id=file_id,
            notifications_cfg=cfg.notifications,
        )

    except Exception as e:
        logger.exception("Job failed with error: %s", e)
        notify_job_status(
            status="failed",
            task_name=task_name_str,
            fm_name=fm_name,
            prep_name=prep_name,
            score_or_error=e,
            file_id=file_id,
            notifications_cfg=cfg.notifications,
        )
        raise e


if __name__ == "__main__":
    main()

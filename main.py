import os
import numpy as np
from resnet_trainer import ResnetTrainer
from vision_transformer_trainer import VisionTransformerTrainer
import optuna
import optunahub
from optuna.study import StudyDirection
from optuna_dashboard import run_server


def objective(trial: optuna.Trial,):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset")
    save_path = os.path.join(BASE_DIR, "saved_models")
    model_name = "vit_64_no_data_augmentation.pth"

    epochs = 30
    batch_size = 64
    img_size = 64
    manual_seed = 42
    only_see_metrics = False
    use_kfold = True
    n_splits = 5
    holdout_test_ratio = 0.10
    stratified_kfold = True
    use_augmentation = False
    num_workers = 1

    if "resnet" in model_name.lower():
        # ResNet settings
        dropout_rate = 0.6
        label_smoothing = 0.05
        weight_decay = 2e-3
        lr_rate = 1e-3
        lr_step_size = 5
        lr_gamma = 0.2
    elif "vit" in model_name.lower():
        # ViT settings
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
        label_smoothing = trial.suggest_float("label_smoothing", 0.01, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1)
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 3e-3, log=True)

        lr_step_size = trial.suggest_int("lr_step_size", 3, 10)
        lr_gamma = trial.suggest_float("lr_gamma", 0.4, 0.9)
        vit_depth = trial.suggest_categorical("vit_depth", [4, 6, 8, 10, 12])
    else:
        raise ValueError(
            "model_name must include either 'resnet' or 'vit' to choose model-specific parameters"
        )
    

    mean_val, std_val = main(dataset_root=root,
        model_name=model_name,
        epochs= epochs,
        lr_rate= lr_rate,
        batch_size=batch_size,
        img_size=img_size,
        manual_seed=manual_seed,
        save_path=save_path,
        only_see_metrics=only_see_metrics,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        use_kfold=use_kfold,
        n_splits=n_splits,
        holdout_test_ratio=holdout_test_ratio,
        stratified_kfold=stratified_kfold,
        use_augmentation=use_augmentation,
        num_workers=num_workers,
        vit_depth=vit_depth)

    return mean_val, std_val

def main(dataset_root: str,
         model_name: str, 
         epochs: int = 5,
         lr_rate: float = 0.01,
         batch_size: int = 32,
         img_size: int = 64, 
         manual_seed: int = 42,
         save_path: str | None = None,
         only_see_metrics: bool = False,
         dropout_rate: float = 0.6,
         label_smoothing: float = 0.05,
         weight_decay: float = 2e-3,
         lr_step_size: int = 5,
         lr_gamma: float = 0.2,
         use_kfold: bool = False,
         n_splits: int = 5,
         holdout_test_ratio: float = 0.15,
         stratified_kfold: bool = True,
         use_augmentation: bool = False,
         num_workers: int = 1,
         vit_depth: int = 6):

    model_name_lower = model_name.lower()
    trainer_class = VisionTransformerTrainer if "vit" in model_name_lower else ResnetTrainer

    trainer = trainer_class(
        dataset_root=dataset_root,
        model_name=model_name,
        epochs=epochs,
        lr_rate=lr_rate,
        batch_size=batch_size,
        img_size=img_size,
        manual_seed=manual_seed,
        save_path=save_path,
        only_see_metrics=only_see_metrics,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        use_kfold=use_kfold,
        n_splits=n_splits,
        holdout_test_ratio=holdout_test_ratio,
        stratified_kfold=stratified_kfold,
        use_augmentation=use_augmentation,
        num_workers=num_workers,
        depth=vit_depth
    )

    if use_kfold:
        fold_best_val_scores = []
        fold_model_paths = []
        
        for fold_idx in range(trainer.fold_count()):
            print(f"\n=== Fold {fold_idx + 1}/{trainer.fold_count()} ===")
            trainer.set_fold(fold_idx)
            trainer.reset_for_new_fold()
            
            # Create fold-specific save path
            fold_save_path = save_path.replace('.pth', f'_fold_{fold_idx+1}.pth')
            trainer.save_path = fold_save_path
            
            trainer.train()

            fold_best_val = max(trainer.val_accuracies) if trainer.val_accuracies else 0.0
            fold_best_val_scores.append(fold_best_val)
            fold_model_paths.append(fold_save_path)
            print(f"Fold {fold_idx + 1} best validation accuracy: {fold_best_val:.2f}%")

        mean_val = float(np.mean(fold_best_val_scores)) if fold_best_val_scores else 0.0
        std_val = float(np.std(fold_best_val_scores)) if fold_best_val_scores else 0.0
        print(f"\nK-Fold best validation accuracy mean: {mean_val:.2f}%")
        print(f"K-Fold best validation accuracy std: {std_val:.2f}%")

        # print("\nRetraining on full trainval set...")
        # trainer.reset_for_new_fold()
        # full_trainval_loader = trainer.build_holdout_trainval_loader(shuffle=True)
        # trainer.trainloader = full_trainval_loader
        
        # # Set final model save path
        # final_save_path = trainer.save_path.replace('.pth', '_final_trainval.pth')
        # trainer.save_path = final_save_path
        
        # trainer.train()

        # print("\nEvaluating final model on holdout test set")
        # trainer.evaluate()
        
        # Save final trained model
        # trainer.save_model(model=trainer.model, save_optimizer=True)
        
        trainer.clear_model()
        print(f"Fold models saved: {fold_model_paths}")
        # print(f"Final model saved: {final_save_path}")

        return mean_val, std_val
    else:
        trainer.train()
        trainer.evaluate()

        trainer.save_model(model=trainer.model, save_optimizer=True)
        trainer.clear_model()

        trainer.plot_metrics()

    
if __name__ == '__main__':
    study_name = "ViT Hyperparameter Study" 
    storage_name = "sqlite:///{}.db".format(study_name)
    module = optunahub.load_module(package="samplers/auto_sampler")
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]

    study = optuna.create_study(
        study_name=study_name,
        sampler=module.AutoSampler(),
        storage=storage_name,
        load_if_exists= True,
        directions=directions
        )

    for _ in range(10):
            study.optimize(objective, n_trials=5)
    print(f"Best value: {study.best_value} (params: {study.best_params})")


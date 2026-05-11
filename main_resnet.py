import os
import numpy as np
from resnet_trainer import ResnetTrainer
from vision_transformer_trainer import VisionTransformerTrainer
import optuna
import optunahub
from optuna.study import StudyDirection
from optuna_dashboard import run_server


def _create_sampler(seed: int = 42) -> optuna.samplers.BaseSampler:
    """Return AutoSampler when available; otherwise use a local fallback sampler."""
    try:
        module = optunahub.load_module(package="samplers/auto_sampler")
        print("Using OptunaHub AutoSampler.")
        return module.AutoSampler()
    except Exception as exc:
        # NSGA-II works well for multi-objective optimization without network access.
        print(f"OptunaHub unavailable ({exc}). Falling back to NSGAIISampler.")
        return optuna.samplers.NSGAIISampler(seed=seed)


def objective(trial: optuna.Trial,):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset_preprocessed")

    model_name = "resnet_9_no_data_augmentation.pth"
    save_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    epochs = 20
    batch_size = 64
    img_size = 64
    manual_seed = 42
    only_see_metrics = False
    use_kfold = True
    n_splits = 5
    test_ratio = 0.10
    stratified_kfold = True
    augment_train_split = False
    augment_test_split = False
    num_workers = 1

    if "resnet" in model_name.lower():
        # ResNet settings
        dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.7, step=0.01)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.01)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 3e-3, log=True)
        lr_step_size = trial.suggest_int("lr_step_size", 4, 12)
        lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.9, step=0.05)
        vit_depth = 6
    elif "vit" in model_name.lower():
        # ViT settings
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.01)
        label_smoothing = trial.suggest_float("label_smoothing", 0.01, 0.3, step=0.01)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, step=0.001)
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 3e-3, log=True, step=0.00001)

        lr_step_size = trial.suggest_int("lr_step_size", 3, 10)
        lr_gamma = trial.suggest_float("lr_gamma", 0.4, 0.9, step=0.01)
        vit_depth = trial.suggest_categorical("vit_depth", [4, 6, 8, 10, 12])
    else:
        raise ValueError(
            "model_name must include either 'resnet' or 'vit' to choose model-specific parameters"
        )

    mean_val, std_val = main(
        dataset_root=root,
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
        test_ratio=test_ratio,
        stratified_kfold=stratified_kfold,
        augment_train_split=augment_train_split,
        augment_test_split=augment_test_split,
        num_workers=num_workers,
        vit_depth=vit_depth,
    )

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
         test_ratio: float = 0.15,
         stratified_kfold: bool = True,
         augment_train_split: bool = False,
         augment_test_split: bool = False,
         num_workers: int = 1,
         vit_depth: int = 6,
         use_val_split: bool = False,
         test_noise_level: str | None = None):

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
        test_ratio=test_ratio,
        stratified_kfold=stratified_kfold,
        augment_train_split=augment_train_split,
        augment_test_split=augment_test_split,
        num_workers=num_workers,
        dataset_is_preprocessed=True,
        depth=vit_depth,
        use_val_split=use_val_split,
        test_noise_level=test_noise_level,
    )

    if use_kfold:
        fold_best_val_scores = []
        fold_model_paths = []

        for fold_idx in range(trainer.fold_count()):
            print(f"\n=== Fold {fold_idx + 1}/{trainer.fold_count()} ===")
            trainer.set_fold(fold_idx)
            trainer.reset_for_new_fold()

            # Create fold-specific save path from current trainer.save_path (always file path).
            base_path = trainer.save_path
            root_path, ext = os.path.splitext(base_path)
            if not ext:
                ext = ".pth"
            fold_save_path = f"{root_path}_fold_{fold_idx + 1}{ext}"
            trainer.save_path = fold_save_path

            trainer.train()

            fold_best_val = max(trainer.test_accuracies) if trainer.test_accuracies else 0.0
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
        if use_val_split:
            trainer.restore_best_model()
        trainer.evaluate()

        mean_val = max(trainer.test_accuracies) if trainer.test_accuracies else 0.0
        std_val = 0
        

        trainer.save_model(model=trainer.model, save_optimizer=True)
        trainer.clear_model()

        # trainer.plot_metrics()
        return mean_val, std_val
    
def run_resnet_augmentation_matrix(
    dataset_root: str,
    save_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    img_size: int = 64,
    manual_seed: int = 42,
    lr_rate: float = 2.0e-3,
    dropout_rate: float = 0.5,
    label_smoothing: float = 0.2,
    weight_decay: float = 1.0e-4,
    lr_step_size: int = 11,
    lr_gamma: float = 0.2,
    test_ratio: float = 0.20,
    num_workers: int = 1,
):
    """Run ResNet with combinations of train augmentation (on/off)
    and test augmentation intensity (none, light, medium, strong).

    Creates descriptive save names and returns a list of result dicts.
    """

    os.makedirs(save_dir, exist_ok=True)
    results = []
    test_levels = ["none", "light", "medium", "strong"]

    for train_aug in (False, True):
        for test_level in test_levels:
            augment_train_split = train_aug
            augment_test_split = False if test_level == "none" else True

            suffix = f"trainAug_{'on' if augment_train_split else 'off'}_test_{test_level}"
            model_name = f"resnet_{suffix}.pth"
            save_path = os.path.join(save_dir, model_name)

            print(f"\n=== Running: {suffix} | save: {save_path} ===")

            mean_val, std_val = main(
                dataset_root=dataset_root,
                model_name=model_name,
                epochs=epochs,
                lr_rate=lr_rate,
                batch_size=batch_size,
                img_size=img_size,
                manual_seed=manual_seed,
                save_path=save_path,
                only_see_metrics=False,
                dropout_rate=dropout_rate,
                label_smoothing=label_smoothing,
                weight_decay=weight_decay,
                lr_step_size=lr_step_size,
                lr_gamma=lr_gamma,
                use_kfold=False,
                n_splits=5,
                test_ratio=test_ratio,
                stratified_kfold=False,
                augment_train_split=augment_train_split,
                augment_test_split=augment_test_split,
                test_noise_level=test_level,
                num_workers=num_workers,
                vit_depth=6,
                use_val_split=False,
            )

            results.append({
                "train_aug": augment_train_split,
                "test_level": test_level,
                "model": save_path,
                "mean_val": mean_val,
                "std_val": std_val,
            })

    return results


if __name__ == '__main__':
    # study_name = "ResNet Hyperparameter Study"
    # storage_name = "sqlite:///{}.db".format(study_name)
    # directions = [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    # sampler = _create_sampler(seed=42)

    # study = optuna.create_study(
    #     study_name=study_name,
    #     sampler=sampler,
    #     storage=storage_name,
    #     load_if_exists=True,
    #     directions=directions
    # )

    # print(study.trials)

    # run_server(storage_name)

    # for _ in range(10):
    #     study.optimize(objective, n_trials=5)


    # for trial in study.best_trials:
    #     print(f"{trial.values} wiht {trial.params}")
    # if study.best_trials:
    #     best_trial = study.best_trials[0]
    #     print(f"Best trial values: {best_trial.values} (params: {best_trial.params})")
    # else:
    #     print("No completed trials yet.")

    # Best params:
    # droput_rate: 0.10
    # label_smoothing: 0.00
    # weight_decay: 8.25e-4
    # lr_rate: 1.2e-3
    # lr_step_size: 9
    # lr_gamma: 0.1

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset_preprocessed")

    model_name = "new_resnet_9_no_data_augment.pth"
    save_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    epochs = 50
    batch_size = 64
    img_size = 64
    manual_seed = 42
    only_see_metrics = True
    use_kfold = False
    n_splits = 5
    test_ratio = 0.20 # Use 80/20 train/test split
    stratified_kfold = False
    augment_train_split = False
    augment_test_split = False
    num_workers = 1

    dropout_rate = 0.5
    label_smoothing = 0.2
    weight_decay = 1.0e-4
    lr_rate = 2.0e-3
    lr_step_size = 11
    lr_gamma = 0.2
    vit_depth = 6
    use_val_split = False 

    # 0.53,0.18,9.652079762372825e-05,0.0020919773917714357,11,0.2

    run_resnet_augmentation_matrix(
        dataset_root=root,
        save_dir=save_dir,
        epochs=epochs,
        lr_rate=lr_rate,
        batch_size=batch_size,
        img_size=img_size,
        manual_seed=manual_seed,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing,
        weight_decay=weight_decay,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        test_ratio=test_ratio,
        num_workers=num_workers,
    )

    # main(
    #     dataset_root=root,
    #     model_name=model_name,
    #     epochs=epochs,
    #     lr_rate=lr_rate,
    #     batch_size=batch_size,
    #     img_size=img_size,
    #     manual_seed=manual_seed,
    #     save_path=save_path,
    #     only_see_metrics=only_see_metrics,
    #     dropout_rate=dropout_rate,
    #     label_smoothing=label_smoothing,
    #     weight_decay=weight_decay,
    #     lr_step_size=lr_step_size,
    #     lr_gamma=lr_gamma,
    #     use_kfold=use_kfold,
    #     n_splits=n_splits,
    #     test_ratio=test_ratio,
    #     stratified_kfold=stratified_kfold,
    #     augment_train_split=augment_train_split,
    #     augment_test_split=augment_test_split,
    #     num_workers=num_workers,
    #     vit_depth=vit_depth,
    #     use_val_split=use_val_split,
    # )

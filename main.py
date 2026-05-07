import os
import csv
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

    model_name = "vit_8_no_data_augmentation.pth"
    save_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    epochs = 100
    batch_size = 64
    img_size = 64
    manual_seed = 42
    only_see_metrics = False
    use_kfold = False
    n_splits = 5
    test_ratio = 0.20
    stratified_kfold = False
    augment_train_split = False
    augment_test_split = False
    num_workers = 1
    embed_dim = 240

    if "resnet" in model_name.lower():
        # ResNet settings
        dropout_rate = 0.6
        label_smoothing = 0.05
        weight_decay = 2e-3
        lr_rate = 1e-3
        lr_step_size = 5
        lr_gamma = 0.2
        vit_depth = 6
    elif "vit" in model_name.lower():
        # ViT settings
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6, step=0.01)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.4, step=0.01)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        lr_rate = trial.suggest_float("lr_rate", 1e-5, 3e-3, log=True)

        lr_step_size = trial.suggest_int("lr_step_size", 3, 12)
        lr_gamma = trial.suggest_float("lr_gamma", 0.3, 0.7, step=0.01)
        vit_depth = trial.suggest_int("vit_depth", 6, 10, step=2)
        embed_dim = trial.suggest_int("vit_embed_dim", 192, 480, step=48)
        num_heads = trial.suggest_int("num_heads", 4, 8, step=2)

    else:
        raise ValueError(
            "model_name must include either 'resnet' or 'vit' to choose model-specific parameters"
        )

    # Print hyperparameters for this trial
    print("\n" + "="*70)
    print("TRIAL HYPERPARAMETERS:")
    print("="*70)
    print(f"dropout_rate: {dropout_rate:.4f}")
    print(f"label_smoothing: {label_smoothing:.4f}")
    print(f"weight_decay: {weight_decay:.6f}")
    print(f"lr_rate: {lr_rate:.8f}")
    print(f"lr_step_size: {lr_step_size}")
    print(f"lr_gamma: {lr_gamma:.4f}")
    print(f"vit_depth: {vit_depth}")
    print(f"vit_embed_dim: {embed_dim}")
    print(f"num_heads: {num_heads}")
    print("="*70 + "\n")

    mean_val = main(
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
        vit_embed_dim=embed_dim,
        vit_depth=vit_depth,
        num_heads=num_heads
    )

    return mean_val


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
         vit_embed_dim: int = 240,
         num_heads: int = 6,
         show_plots: bool = True,
         use_val_split: bool = False):

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
        embed_dim=vit_embed_dim,
        num_heads=num_heads,
        use_val_split=use_val_split
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

        return mean_val
    else:
        trainer.train()
        if use_val_split:
            trainer.restore_best_model()
        trainer.evaluate()

        result = {
            "best_train_acc": max(trainer.train_accuracies) if trainer.train_accuracies else 0.0,
            "best_val_acc": max(trainer.val_accuracies) if trainer.val_accuracies else 0.0,
            "test_acc": trainer.test_accuracy if trainer.test_accuracy is not None else 0.0,
            "test_loss": trainer.test_loss if trainer.test_loss is not None else float("nan"),
        }

        if use_val_split:
            mean_val = result["best_val_acc"]
        else:
            mean_val = result["test_acc"]
        

        # trainer.save_model(model=trainer.model, save_optimizer=True)
        trainer.clear_model()

        # trainer.save_model(model=trainer.model, save_optimizer=True)
        # if show_plots:
        #     trainer.plot_metrics()
        # trainer.clear_model()
        # return result

        return mean_val

if __name__ == '__main__':
    # study_name = "ViT New New Hyperparameter Study"
    # storage_name = "sqlite:///{}.db".format(study_name)
    # directions = [StudyDirection.MAXIMIZE]
    # sampler = _create_sampler(seed=42)

    # study = optuna.create_study(
    #     study_name=study_name,
    #     sampler=sampler,
    #     storage=storage_name,
    #     load_if_exists=True,
    #     directions=directions
    # )


    # # run_server(storage_name)

    # # print(study.trials)

    # for _ in range(60):
    #     study.optimize(objective, n_trials=1)

    # for trial in study.best_trials:
    #     print(f"{trial.values} wiht {trial.params}")
    # if study.best_trials:
    #     best_trial = study.best_trials[0]
    #     print(f"Best trial values: {best_trial.values} (params: {best_trial.params})")
    # else:
    #     print("No completed trials yet.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(BASE_DIR, "dataset_preprocessed")

    model_name = "vit_8_no_data_augmentation.pth"
    save_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    epochs = 100
    batch_size = 64
    img_size = 64
    manual_seed = 42
    only_see_metrics = False
    use_kfold = False
    n_splits = 5
    test_ratio = 0.20
    stratified_kfold = True
    augment_train_split = True
    augment_test_split = True
    num_workers = 1

    dropout_rate = 0.2
    label_smoothing = 0.25
    weight_decay = 0.001
    lr_rate = 2e-4
    lr_step_size = 9
    lr_gamma = 0.7
    vit_depth = 6
    vit_embed_dim = 192
    num_heads = 8
    use_val_split = False


    # # # Show transformations
    # from data_processing.data_augmentation import DataAugmentation

    # aug = DataAugmentation(img_size=64, output_channels=1)
    # aug.showAugmentedSamples(
    # image_path="dataset_preprocessed/banana/0000.jpg",
    # horizontal_flip=True,
    # vertical_flip=False,
    # translation=True,
    # blur=False,
    # random_erasing=False,
    # noise_light=True,
    # noise_medium=True,
    # noise_strong=True,
    # scale=True,
    # preprocessed_input=True,
    # include_base=True,
    # )

    main(
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
        vit_embed_dim=vit_embed_dim,
        num_heads=num_heads,
        show_plots=True,
        use_val_split=use_val_split,
    )

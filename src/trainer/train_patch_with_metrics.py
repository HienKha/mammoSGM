import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import torch
import csv
from src.data.patch_dataset import get_dataloaders
from src.data.dataloader import load_metadata
from src.trainer.engines import train_model, evaluate_model_and_save_metrics
from src.utils.common import load_config, get_arg_or_config, clear_cuda_memory
from src.trainer.train_based import parse_img_size



def prepare_data_and_model(
    data_folder,
    model,
    batch_size,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    pretrained_model_path=None,
    img_size=None,
    target_column=None,  # thêm target_column
):
    clear_cuda_memory()
    train_df, test_df, _ = load_metadata(
        data_folder, config_path, target_column=target_column
    )
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        num_patches=num_patches,
        img_size=img_size,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained_model_path:
        try:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠️ Error loading pretrained model: {e}. Training from scratch.")
    model = model.to(device)
    return train_df, test_df, train_loader, test_loader, model, device


def run_train(
    data_folder,
    model,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    patience=50,
    loss_type="ce",
    model_type=None,
    img_size=None,
    pretrained_model_path=None,
    target_column=None,  # thêm target_column
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            data_folder,
            model,
            batch_size,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
            pretrained_model_path=pretrained_model_path,
            img_size=img_size,
            target_column=target_column,
        )
    )
    model_name = f"{model_type}" if model_type else arch_type
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_name,
        output=output,
        dataset_folder=data_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
        arch_type=arch_type,
        num_patches=num_patches,  # truyền số patch để lưu tên file đúng
    )
    return trained_model


def run_test(
    data_folder,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    pretrained_model_path=None,
    img_size=None,
    target_column=None,  # thêm target_column
    setting=None,
    backbone_name=None
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        model,
        batch_size,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
        pretrained_model_path=pretrained_model_path,
        img_size=img_size,
        target_column=target_column,
    )
    print("\nEvaluation on Test Set:")
    # test_loss, test_acc = evaluate_model(
    #     model, test_loader, device=device, mode="Test", return_loss=True
    # )
    metrics = evaluate_model_and_save_metrics(
        model, test_loader, device="cpu", mode="Test", output_dir=None, model_info=None, criterion=None
    )
    dataset = data_folder.split('/')[-1]
    # return test_loss, test_acc
    target_column = target_column if target_column else "target"
    pretrained_model_path = pretrained_model_path if pretrained_model_path else "model"
    setting = setting if setting else "default"

    metrics.update({
        "dataset": dataset,
        "backbone": backbone_name,
        "target_column": target_column,
        "setting": setting,
    })

    saved_csv_file = os.path.join(output, "test_metrics_MIL.csv")

    if os.path.isfile(saved_csv_file):
        with open(saved_csv_file, 
                  'a') as f:
            w = csv.DictWriter(f, metrics.keys())
            w.writerow(metrics)
    else:
        with open(saved_csv_file, 
                  'w') as f:
            w = csv.DictWriter(f, metrics.keys())
            w.writeheader()
            w.writerow(metrics)


    # save the metrics dictionary as a csv file
    # metrics_df = pd.DataFrame([metrics])
    print(f"Saved test metrics to {saved_csv_file}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder",
    )
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--loss_type", type=str, choices=["ce", "focal"])
    parser.add_argument("--num_patches", type=int)
    parser.add_argument(
        "--arch_type",
        type=str,
        default="patch_resnet",
        choices=[
            "patch_resnet",
            "patch_transformer",
            "token_mixer",
            "global_local",
            "global_local_token",
            "mil",
            "mil_v2",
            "mil_v3",
            "mil_v4",
            "mil_v5",
            "mil_v6",
            "mil_v7",
            "mil_v8",
            "mil_v9",
            "mil_v10",  # added v10
            "mil_v11",  # added v11
            "mil_v12",  # added v12
        ],
    )
    parser.add_argument("--img_size", type=str, default=None)
    parser.add_argument(
        "--target_column", type=str, default=None, help="Name of target column"
    )
    parser.add_argument(
        "--setting", type=str, default=None, help="Setting name based or MIL"
    )
    parser.add_argument(
        "--backbone_name", type=str, default=None, help="Backbone name resnet34, convnextv2_tiny, resnet50"
    ) 

    args = parser.parse_args()
    config = load_config(args.config)

    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
    model_type = get_arg_or_config(args.model_type, config.get("model_type"), None)
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    num_epochs = get_arg_or_config(args.num_epochs, config.get("num_epochs"), 10)
    lr = get_arg_or_config(args.lr, config.get("lr"), 1e-4)
    pretrained_model_path = get_arg_or_config(
        args.pretrained_model_path, config.get("pretrained_model_path"), None
    )
    output = get_arg_or_config(args.output, config.get("output"), "output")
    patience = get_arg_or_config(args.patience, config.get("patience"), 50)
    loss_type = get_arg_or_config(args.loss_type, config.get("loss_type"), "ce")
    num_patches = get_arg_or_config(args.num_patches, config.get("num_patches"), 2)
    arch_type = get_arg_or_config(
        args.arch_type, config.get("arch_type"), "patch_resnet"
    )
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    target_column = get_arg_or_config(
        args.target_column, config.get("target_column"), None
    )
    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)
    setting = get_arg_or_config(
        args.setting, config.get("setting"), None
    )
    backbone_name = get_arg_or_config(
        args.backbone_name, config.get("backbone_name"), None
    )   

    # model_type = "vit_small"
    # arch_type = "patch_transformer"

    from src.models.patch_model import get_patch_model

    train_df, test_df, class_names = load_metadata(
        data_folder, args.config, target_column=target_column, print_stats=False
    )
    

    model = get_patch_model(
        model_type=model_type,
        num_patches=num_patches,
        arch_type=arch_type,
        num_classes=len(class_names),  # assuming binary classification; adjust as needed
    )

    if args.mode == "train":
        run_train(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            output=output,
            config_path=args.config,
            num_patches=num_patches,
            arch_type=arch_type,
            patience=patience,
            loss_type=loss_type,
            model_type=model_type,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
    elif args.mode == "test":
        run_test(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            num_patches=num_patches,
            arch_type=arch_type,
            pretrained_model_path=pretrained_model_path,
            img_size=img_size,
            target_column=target_column,
            setting=setting,
            backbone_name=backbone_name,
        )

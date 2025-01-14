import torch
import numpy as np
from tqdm import tqdm
from model import Classifier
from dataset import ImageNetteDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def load_validation_dataset(path, batch_size=16):
    """Carga el conjunto de validación con transformaciones predefinidas."""
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
    ])
    val_dataset = ImageNetteDataset(path, split='val', transform=transform_val)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def bootstrap_uncertainty(model, dataloader, device, n_iterations=1000):
    """Realiza un análisis de incertidumbre basado en bootstrap."""
    model.eval()
    top1_accuracies, top5_accuracies = [], []

    # Realizar N iteraciones de bootstrap
    for _ in tqdm(range(n_iterations), desc="Bootstrap Iterations"):
        correct_top1, correct_top5, total = 0, 0, 0
        with torch.no_grad():
            # Iterar sobre el dataloader y realizar muestreo aleatorio con reemplazo
            for x, y in dataloader:
                indices = np.random.choice(len(x), len(x), replace=True)  # Muestreo con reemplazo
                x, y = x[indices].to(device), y[indices].to(device)

                _, pred = model(x)

                # Top-1 accuracy
                predicted_class = pred.argmax(dim=1)
                correct_top1 += (predicted_class == y).sum().item()

                # Top-5 accuracy
                top5_pred = torch.topk(pred, 5, dim=1).indices
                correct_top5 += sum([y[i] in top5_pred[i] for i in range(len(y))])

                total += y.numel()

        # Calcular precisión para esta iteración
        top1_accuracies.append(correct_top1 / total)
        top5_accuracies.append(correct_top5 / total)

    # Calcular intervalos de confianza
    top1_ci = (np.percentile(top1_accuracies, 2.5), np.percentile(top1_accuracies, 97.5))
    top5_ci = (np.percentile(top5_accuracies, 2.5), np.percentile(top5_accuracies, 97.5))

    return top1_ci, top5_ci


def main():
    # Configurar dispositivo y ruta del modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "train_log/20250111_1249/best_model.pth"
    val_dataset_path = "data/imagenette2-320"

    # Cargar el modelo -> cargamos el small (num_ViGBlocks = depth)
    model = Classifier(
        n_classes=10, num_ViGBlocks=12, out_feature=320, num_edges=10, head_num=2, patch_size=8
    )
    # model = Classifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Cargar el conjunto de validación
    val_dataloader = load_validation_dataset(val_dataset_path, batch_size=8)

    # Realizar análisis de incertidumbre
    top1_ci, top5_ci = bootstrap_uncertainty(model, val_dataloader, device)

    print(f"Top-1 Confidence Interval: {top1_ci}")
    print(f"Top-5 Confidence Interval: {top5_ci}")


if __name__ == "__main__":
    main()

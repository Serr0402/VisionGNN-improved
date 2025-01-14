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


def propagate_uncertainty(model, dataloader, device, perturbation_rate=0.1, n_iterations=3):
    """Realiza un análisis de propagación de incertidumbre en las predicciones."""
    model.eval()
    perturbed_top1, perturbed_top5 = [], []

    for _ in tqdm(range(n_iterations), desc="Propagation Iterations"):
        correct_top1, correct_top5, total = 0, 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                _, pred = model(x)

                # Perturbar las predicciones finales
                predicted_class = pred.argmax(dim=1)
                top5_pred = torch.topk(pred, 5, dim=1).indices

                # Introducir ruido en las etiquetas
                perturbed_y = y.clone()
                perturb_indices = np.random.choice(
                    len(perturbed_y),
                    int(len(perturbed_y) * perturbation_rate),
                    replace=False
                )
                for idx in perturb_indices:
                    perturbed_y[idx] = torch.randint(0, pred.size(1), (1,), device=device).item()

                # Top-1 accuracy con perturbaciones
                correct_top1 += (predicted_class == perturbed_y).sum().item()

                # Top-5 accuracy con perturbaciones
                correct_top5 += sum([perturbed_y[i] in top5_pred[i] for i in range(len(perturbed_y))])

                total += perturbed_y.numel()

        # Calcular precisión para esta iteración
        perturbed_top1.append(correct_top1 / total)
        perturbed_top5.append(correct_top5 / total)

    # Calcular estadísticas
    top1_mean, top1_std = np.mean(perturbed_top1), np.std(perturbed_top1)
    top5_mean, top5_std = np.mean(perturbed_top5), np.std(perturbed_top5)

    return (top1_mean, top1_std), (top5_mean, top5_std)


def main():
    # Configurar dispositivo y ruta del modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "train_log/20250111_1249/best_model.pth"
    val_dataset_path = "data/imagenette2-320"

    # Cargar el modelo -> cargamos el small (num_ViGBlocks = depth)
    model = Classifier(
        n_classes=10, num_ViGBlocks=12, out_feature=320, num_edges=10, head_num=2, patch_size=8
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Cargar el conjunto de validación
    val_dataloader = load_validation_dataset(val_dataset_path, batch_size=8)

    # Realizar análisis de propagación de incertidumbre
    perturbation_rate = 0.2  # 20% de etiquetas perturbadas
    n_iterations = 500  # Número de iteraciones
    top1_stats, top5_stats = propagate_uncertainty(model, val_dataloader, device, perturbation_rate, n_iterations)

    print(f"Top-1 Mean: {top1_stats[0]:.4f}, Std Dev: {top1_stats[1]:.4f}")
    print(f"Top-5 Mean: {top5_stats[0]:.4f}, Std Dev: {top5_stats[1]:.4f}")


if __name__ == "__main__":
    main()

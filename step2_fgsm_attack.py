"""
Step 2: FGSM (Fast Gradient Sign Method) attack on the trained MNIST classifier.

Idea: find the direction that increases the model's loss the fastest (the
gradient of the loss w.r.t. the input image), then push every pixel a tiny
step in that direction. One step only.

adv_image = clean_image + epsilon * sign(gradient of loss w.r.t. clean_image)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from step1_train_classifier import SimpleCNN, MODEL_PATH, DEVICE

BATCH_SIZE = 128
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 0.0 = clean accuracy (no attack)
NUM_SAMPLE_IMAGES = 6  # how many before/after images to save for the report


def get_test_loader():
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def load_model():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def fgsm_attack(model, images, labels, epsilon):
    """Craft adversarial images with a single gradient step."""
    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Direction that increases the loss the fastest, at each pixel.
    grad_sign = images.grad.data.sign()

    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)  # keep valid pixel range [0, 1]
    return adv_images.detach()


@torch.no_grad()
def _accuracy_on(model, images, labels):
    preds = model(images).argmax(dim=1)
    return (preds == labels).sum().item()


def evaluate_fgsm(model, test_loader, epsilon):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if epsilon == 0.0:
            correct += _accuracy_on(model, images, labels)
        else:
            adv_images = fgsm_attack(model, images, labels, epsilon)
            correct += _accuracy_on(model, adv_images, labels)

        total += labels.size(0)
    return correct / total


def save_sample_images(model, test_loader, epsilon=0.2):
    """Save a few clean vs adversarial images side by side, for the report."""
    images, labels = next(iter(test_loader))
    images, labels = images[:NUM_SAMPLE_IMAGES], labels[:NUM_SAMPLE_IMAGES]
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    adv_images = fgsm_attack(model, images, labels, epsilon)

    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    print("\nSample predictions (label -> clean_pred -> adv_pred):")
    for i in range(NUM_SAMPLE_IMAGES):
        print(f"  true={labels[i].item()}  clean_pred={clean_preds[i].item()}  "
              f"adv_pred={adv_preds[i].item()}  {'FOOLED' if adv_preds[i] != labels[i] else 'still correct'}")

    save_image(images, "fgsm_clean_samples.png", nrow=NUM_SAMPLE_IMAGES)
    save_image(adv_images, "fgsm_adversarial_samples.png", nrow=NUM_SAMPLE_IMAGES)
    print(f"\nSaved fgsm_clean_samples.png and fgsm_adversarial_samples.png (epsilon={epsilon})")


def main():
    print(f"Using device: {DEVICE}")
    model = load_model()
    test_loader = get_test_loader()

    print("\nepsilon | accuracy")
    print("--------|---------")
    for eps in EPSILONS:
        acc = evaluate_fgsm(model, test_loader, eps)
        print(f"{eps:>7.2f} | {acc * 100:6.2f}%")

    save_sample_images(model, test_loader, epsilon=0.2)


if __name__ == "__main__":
    main()

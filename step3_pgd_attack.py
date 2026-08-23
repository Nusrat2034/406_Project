"""
Step 3: PGD (Projected Gradient Descent) attack on the trained MNIST classifier.

Idea: FGSM but repeated many times with a small step size. After each small
step, "project" the adversarial image back so it never strays farther than
epsilon from the original clean image (this is the "projected" part).

Loop (repeated num_steps times):
    adv_image = adv_image + alpha * sign(gradient of loss w.r.t. adv_image)
    adv_image = clip(adv_image, clean_image - epsilon, clean_image + epsilon)
    adv_image = clip(adv_image, 0, 1)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from step1_train_classifier import SimpleCNN, MODEL_PATH, DEVICE
from step2_fgsm_attack import evaluate_fgsm  # reuse for the FGSM comparison column

BATCH_SIZE = 128
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
PGD_STEPS = 20
PGD_ALPHA = 0.01  # step size per iteration
NUM_SAMPLE_IMAGES = 6


def get_test_loader():
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def load_model():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def pgd_attack(model, images, labels, epsilon, alpha=PGD_ALPHA, num_steps=PGD_STEPS):
    """Craft adversarial images with repeated small gradient steps."""
    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)

    adv_images = images.clone().detach()

    for _ in range(num_steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad_sign = adv_images.grad.data.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back into the epsilon-ball around the original image.
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()

    return adv_images


@torch.no_grad()
def _accuracy_on(model, images, labels):
    preds = model(images).argmax(dim=1)
    return (preds == labels).sum().item()


def evaluate_pgd(model, test_loader, epsilon):
    if epsilon == 0.0:
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += _accuracy_on(model, images, labels)
            total += labels.size(0)
        return correct / total

    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        adv_images = pgd_attack(model, images, labels, epsilon)
        correct += _accuracy_on(model, adv_images, labels)
        total += labels.size(0)
    return correct / total


def save_sample_images(model, test_loader, epsilon=0.2):
    images, labels = next(iter(test_loader))
    images, labels = images[:NUM_SAMPLE_IMAGES], labels[:NUM_SAMPLE_IMAGES]
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    adv_images = pgd_attack(model, images, labels, epsilon)

    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    print("\nSample predictions (label -> clean_pred -> adv_pred):")
    for i in range(NUM_SAMPLE_IMAGES):
        print(f"  true={labels[i].item()}  clean_pred={clean_preds[i].item()}  "
              f"adv_pred={adv_preds[i].item()}  {'FOOLED' if adv_preds[i] != labels[i] else 'still correct'}")

    save_image(adv_images, "pgd_adversarial_samples.png", nrow=NUM_SAMPLE_IMAGES)
    print(f"\nSaved pgd_adversarial_samples.png (epsilon={epsilon}, steps={PGD_STEPS})")


def main():
    print(f"Using device: {DEVICE}")
    print(f"PGD settings: steps={PGD_STEPS}, alpha={PGD_ALPHA}")
    model = load_model()
    test_loader = get_test_loader()

    print("\nepsilon | FGSM accuracy | PGD accuracy")
    print("--------|----------------|-------------")
    for eps in EPSILONS:
        fgsm_acc = evaluate_fgsm(model, test_loader, eps)
        pgd_acc = evaluate_pgd(model, test_loader, eps)
        print(f"{eps:>7.2f} | {fgsm_acc * 100:13.2f}% | {pgd_acc * 100:11.2f}%")

    save_sample_images(model, test_loader, epsilon=0.2)


if __name__ == "__main__":
    main()

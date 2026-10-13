import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 preview PNG files.")
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--output-dir", default="../../data/cifar10_preview")
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()

    dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {class_name: 0 for class_name in dataset.classes}
    for image, label in dataset:
        class_name = dataset.classes[label]
        if counts[class_name] >= args.per_class:
            continue
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image.save(class_dir / f"{class_name}_{counts[class_name]:02d}.png")
        counts[class_name] += 1
        if all(count >= args.per_class for count in counts.values()):
            break

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
    print(f"Preview images written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

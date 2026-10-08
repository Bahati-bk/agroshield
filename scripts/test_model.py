import torch

from src.agroshield.training.model import (
    CLASS_NAMES,
    create_model,
    count_parameters,
    model_summary,
)


def main() -> None:
    print("=" * 70)
    print("AGROSHIELD — MODEL DEFINITION TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Create model
    # ---------------------------------------------------------------
    print("\n[1] Creating MobileNetV3-Small")
    print("-" * 60)

    model = create_model(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
    )

    print(f"Model: MobileNetV3-Small")
    print(f"Classes: {len(CLASS_NAMES)}")

    # ---------------------------------------------------------------
    # 2. Parameter count
    # ---------------------------------------------------------------
    print("\n[2] Parameter count")
    print("-" * 60)

    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    assert total_parameters > 0
    assert trainable_parameters > 0

    # ---------------------------------------------------------------
    # 3. Verify classifier
    # ---------------------------------------------------------------
    print("\n[3] Classifier validation")
    print("-" * 60)

    classifier_output_features = model.classifier[-1].out_features

    print(
        f"Classifier output features: "
        f"{classifier_output_features}"
    )

    assert classifier_output_features == len(CLASS_NAMES)

    print("Classifier: PASSED")

    # ---------------------------------------------------------------
    # 4. Forward-pass smoke test
    # ---------------------------------------------------------------
    print("\n[4] Forward-pass smoke test")
    print("-" * 60)

    batch_size = 2

    dummy_input = torch.randn(
        batch_size,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")

    expected_output_shape = (batch_size, len(CLASS_NAMES))

    assert output.shape == expected_output_shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    print("Forward pass: PASSED")

    # ---------------------------------------------------------------
    # 5. Class vocabulary
    # ---------------------------------------------------------------
    print("\n[5] Class vocabulary")
    print("-" * 60)

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")

    assert len(CLASS_NAMES) == 5
    assert len(set(CLASS_NAMES)) == 5

    print("Class vocabulary: PASSED")

    # ---------------------------------------------------------------
    # 6. Model summary
    # ---------------------------------------------------------------
    print("\n[6] Model summary")
    print("-" * 60)

    summary = model_summary(model)

    for key, value in summary.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------------
    # Complete
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL DEFINITION TEST COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
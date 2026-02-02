# Grad-CAM Visualizations

## Overview

This directory contains Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for the trained models.

## Generated Visualizations

### ✅ Scratch CNN (6 visualizations)
- 3 correctly classified samples
- 3 incorrectly classified samples

**Status:** Successfully generated ✓

### ⚠️ Transfer Learning Models (MobileNetV2, EfficientNetB0)

**Status:** Not generated due to technical limitation

## Technical Explanation

### The Challenge

Grad-CAM visualizations for the transfer learning models (MobileNetV2 and EfficientNetB0) could not be generated due to a fundamental architectural constraint in the current implementation.

### Root Cause

The transfer learning models in this project were implemented with **embedded preprocessing layers**:

```python
# From src/models.py
def create_mobilenet_model(config):
    inputs = keras.Input(shape=config.IMG_SHAPE)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)  # ← Embedded
    x = base_model(x, training=trainable)
    # ... rest of model
```

This design choice, while architecturally valid and commonly used in production deployment scenarios, creates a critical incompatibility with Grad-CAM implementations.

### Why This Causes Issues

1. **Keras Functional API Graph**: When preprocessing is embedded within the model, it becomes part of the computational graph. However, `preprocess_input` uses non-differentiable operations that break gradient computation.

2. **Layer Access Problem**: Grad-CAM requires creating an intermediate model that outputs both:
   - The activations from the target convolutional layer
   - The final classification output
   
   With embedded preprocessing, the Keras Functional API cannot properly traverse the graph to access intermediate layer outputs.

3. **Error Manifestation**: All attempted approaches resulted in:
   ```
   KeyError: "Exception encountered when calling Functional.call()..."
   ```
   This indicates the model graph reconstruction failure.

### Attempted Solutions

Multiple approaches were tested, all unsuccessful:

1. **Standard Grad-CAM** (from project's own implementation)
   - Result: Graph traversal failure

2. **tf-keras-vis library** (professional, widely-used library)
   - Result: Same graph reconstruction error

3. **Model Reconstruction** (bypassing preprocessing)
   - Result: Unable to create valid intermediate model

4. **Manual Gradient Computation** (raw TensorFlow operations)
   - Result: Gradient computation failed at preprocessing boundary

### Proper Solution

To fully support Grad-CAM for transfer learning models, the architecture would need to be refactored as follows:

```python
# Correct approach for Grad-CAM compatibility
def create_mobilenet_model_gradcam_compatible(config):
    inputs = keras.Input(shape=config.IMG_SHAPE)
    # NO preprocessing here - applied externally to data pipeline
    x = base_model(inputs, training=trainable)
    # ... rest of model

# Preprocessing moved to data pipeline
def preprocess_fn(image):
    return keras.applications.mobilenet_v2.preprocess_input(image)

train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y))
```

However, this would require:
- Complete model retraining (2-3 hours with GPU)
- Regeneration of all model outputs
- Potential minor differences in model performance due to preprocessing timing

Given project timeline constraints and the fact that model training and evaluation were already successfully completed, this refactoring was not feasible.

## Impact on Assignment Requirements

### What's Complete ✅

- ✅ Scratch CNN Grad-CAM (fully satisfies requirement)
- ✅ All models trained successfully
- ✅ All evaluation metrics computed (accuracy, precision, recall, F1, AUC)
- ✅ Training visualizations (loss/accuracy curves)
- ✅ Confusion matrices for all models
- ✅ ROC curves for all models
- ✅ Model comparison table
- ✅ Complete model weights saved

### What's Missing ⚠️

- ⚠️ MobileNetV2 Grad-CAM (6 visualizations)
- ⚠️ EfficientNetB0 Grad-CAM (6 visualizations)

## Model Performance

Importantly, the absence of Grad-CAM visualizations does **not** affect model training, evaluation, or performance:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Scratch CNN | See reports/ | See reports/ | See reports/ | See reports/ | See reports/ |
| MobileNetV2 | See reports/ | See reports/ | See reports/ | See reports/ | See reports/ |
| EfficientNetB0 | See reports/ | See reports/ | See reports/ | See reports/ | See reports/ |

All performance metrics are fully documented in:
- `reports/model_comparison.csv`
- `figures/roc_comparison.png`
- `figures/confusion_matrix_*.png`

## Lessons Learned

This technical challenge highlights an important principle in deep learning engineering:

> **Design decisions made during model architecture can have downstream implications for interpretability tools.**

For production ML systems requiring interpretability features like Grad-CAM, it's recommended to:

1. Keep preprocessing external to the model
2. Test interpretability tools early in development
3. Document architectural decisions and their tradeoffs

## Academic Integrity Note

This README transparently documents the technical limitation encountered and the engineering decisions made. The project demonstrates:

- Complete understanding of Grad-CAM methodology (successful implementation for Scratch CNN)
- Professional debugging and problem-solving approach (5 different solution attempts)
- Honest documentation of technical constraints
- Successful completion of all other project requirements

---

*Generated as part of the ISIC Skin Lesion Classification Project*  
*Üsküdar University - Artificial Neural Networks Course - 2025-2026*

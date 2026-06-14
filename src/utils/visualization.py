"""
Publication-ready visualization functions.
All plots: 300 DPI, tight layout, clear labels.
Font: serif family for IEEE paper compatibility.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# IEEE-friendly defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

SAVE_DIR = "results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_actual_vs_predicted(y_true, y_pred, model_name, product_name, filename):
    """Line chart of actual vs predicted sales over time.

    Args:
        y_true: Array of actual sales values.
        y_pred: Array of predicted sales values.
        model_name: Name of the model (shown in legend and title).
        product_name: Product family name (shown in title).
        filename: Output filename without extension (saved to results/plots/).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label='Actual', color='#1f77b4', linewidth=1.5)
    ax.plot(y_pred, label=f'{model_name} Predicted', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Sales (Units)')
    ax.set_title(f'Actual vs Predicted — {model_name} ({product_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_metric_comparison_bar(model_names, metric_values, metric_name, filename):
    """Bar chart comparing one metric across models.

    Args:
        model_names: List of model name strings.
        metric_values: List of metric values corresponding to each model.
        metric_name: Display name for the metric (axis label and title).
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(model_names, metric_values, color=colors[:len(model_names)],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison Across Models')
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_model_size_vs_accuracy(model_names, sizes_kb, accuracies, filename):
    """Scatter plot showing model size vs accuracy trade-off.

    Args:
        model_names: List of model name strings.
        sizes_kb: List of model sizes in KB.
        accuracies: List of MAPE values (lower is better).
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    for i, name in enumerate(model_names):
        ax.scatter(sizes_kb[i], accuracies[i], s=120, c=colors[i % len(colors)],
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(name, (sizes_kb[i], accuracies[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    ax.set_xlabel('Model Size (KB)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Model Size vs Prediction Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_confusion_matrix(cm, labels, model_name, filename):
    """Heatmap confusion matrix.

    Args:
        cm: 2D numpy confusion matrix array.
        labels: List of class label strings.
        model_name: Name of the model (shown in title).
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_training_loss(train_losses, val_losses, model_name, filename):
    """Training and validation loss curves over epochs.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: List of validation loss values per epoch.
        model_name: Name of the model (shown in title).
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Training Loss', color='#2ecc71', linewidth=1.5)
    ax.plot(val_losses, label='Validation Loss', color='#e74c3c', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'Training Convergence — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_per_language_accuracy(model_names, lang_accuracies, filename):
    """Grouped bar chart showing accuracy per language per model.

    Args:
        model_names: List of model name strings.
        lang_accuracies: Dict like {'English': [acc1, acc2, ...], 'Hindi': [...], ...}
                         where each list has one value per model.
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.2
    colors = {'English': '#3498db', 'Hindi': '#e74c3c', 'Kannada': '#2ecc71', 'Code-Mixed': '#f39c12'}

    for i, (lang, accs) in enumerate(lang_accuracies.items()):
        bars = ax.bar(x + i * width, accs, width, label=lang,
                      color=colors.get(lang, '#999'), edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Intent Classification Accuracy by Language')
    ax.set_xticks(x + width * (len(lang_accuracies) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_compression_comparison(model_names, full_sizes, compressed_sizes, filename):
    """Side-by-side bar chart comparing full vs compressed model sizes.

    Args:
        model_names: List of model name strings.
        full_sizes: List of full model sizes in KB.
        compressed_sizes: List of compressed model sizes in KB.
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.3
    ax.bar(x - width / 2, full_sizes, width, label='Full Model',
           color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, compressed_sizes, width, label='Compressed (TFLite/ONNX)',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Model')
    ax.set_ylabel('Size (KB)')
    ax.set_title('Model Compression: Full vs Edge-Optimized')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()


def plot_per_intent_f1(model_names, intent_f1s, intents, filename):
    """Grouped bar chart showing per-intent F1 score across models.

    Args:
        model_names: List of model name strings.
        intent_f1s: Dict like {'check_stock': [f1_m1, f1_m2, ...], ...}
        intents: List of intent label strings.
        filename: Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(intents))
    n = len(model_names)
    width = 0.8 / n
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for i, model in enumerate(model_names):
        vals = [intent_f1s.get(intent, [0] * n)[i] for intent in intents]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=model,
               color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Intent')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Intent F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(intents, rotation=15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/{filename}.png")
    plt.close()

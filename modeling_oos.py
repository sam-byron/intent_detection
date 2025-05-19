import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def max_softmax_prob(logits: torch.Tensor, T: float = 1.0) -> float:
    probs = F.softmax(logits / T, dim=-1)
    return float(probs.max())

def energy_score(logits: torch.Tensor, T: float = 1.0) -> float:
    return float(-T * torch.logsumexp(logits / T, dim=-1))

def calibrate_oos_thresholds_roc(
    trainer,
    val_dataset,
    oos_label_name: str = "oos:oos",
    T: float = 1.0,
    plot: bool = False
):
    # 1. Gather logits & labels
    pred_output = trainer.predict(val_dataset)
    logits_np   = pred_output.predictions    # (N, num_labels)
    labels_np   = pred_output.label_ids      # (N,)

    # 2. Identify OOS ID
    label2id = trainer.model.config.label2id
    if oos_label_name not in label2id:
        raise ValueError(f"Label '{oos_label_name}' not found.")
    oos_id = label2id[oos_label_name]

    # 3. Compute scores & ground-truth
    max_probs = []
    energies  = []
    is_oos    = []
    for logit_vec, lbl in zip(torch.tensor(logits_np), labels_np):
        max_probs.append(max_softmax_prob(logit_vec, T))
        energies.append(energy_score(logit_vec, T))
        is_oos.append(int(lbl) == oos_id)

    max_probs = np.array(max_probs)
    energies  = np.array(energies)
    is_oos    = np.array(is_oos, dtype=bool)

    # Optional: visualize
    if plot:
        plt.hist(max_probs[~is_oos], bins=50, alpha=0.5, label="in-scope")
        plt.hist(max_probs[is_oos],  bins=50, alpha=0.5, label="OOS")
        plt.title("Softmax Max-Prob Distributions"); plt.legend(); plt.show()

        plt.hist(energies[~is_oos], bins=50, alpha=0.5, label="in-scope")
        plt.hist(energies[is_oos],  bins=50, alpha=0.5, label="OOS")
        plt.title("Energy Score Distributions"); plt.legend(); plt.show()

    # 4. Find best softmax-prob threshold via Youden’s J
    #    We invert max_probs so that higher “score” => more likely OOS
    fpr_p, tpr_p, thr_p = roc_curve(is_oos, -max_probs)
    j_scores_p = tpr_p - fpr_p
    best_idx_p = np.argmax(j_scores_p)
    tau_prob   = thr_p[best_idx_p]

    # 5. Find best energy threshold via Youden’s J
    fpr_e, tpr_e, thr_e = roc_curve(is_oos, energies)
    j_scores_e = tpr_e - fpr_e
    best_idx_e = np.argmax(j_scores_e)
    tau_energy = thr_e[best_idx_e]

    print(f"→ Softmax-prob τₚ = {tau_prob:.4f} (J={j_scores_p[best_idx_p]:.3f})")
    print(f"→ Energy τ_E        = {tau_energy:.3f} (J={j_scores_e[best_idx_e]:.3f})")

    # 6. Evaluate combined detector
    preds_oos = (max_probs < tau_prob) | (energies > tau_energy)
    tn, fp, fn, tp = confusion_matrix(is_oos, preds_oos).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("Combined Detector — Confusion Matrix:")
    print(f"    True In-Scope   as In-Scope:  TN={tn}")
    print(f"    True In-Scope   as OOS:       FP={fp}")
    print(f"    True OOS        as In-Scope:  FN={fn}")
    print(f"    True OOS        as OOS:       TP={tp}")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    return tau_prob, tau_energy


# 3️⃣ Augmented inference function with OOS/OOD detection
def predict_intent_with_oos(
    text: str,
    tau_prob: float,
    tau_energy: float,
    T: float = 1.0
) -> dict:
    """
    Tokenize `text`, run the model, and:
      - compute max softmax prob & energy,
      - if either indicates OOD, return 'oos',
      - otherwise return the predicted in-scope intent.
    Returns a dict with:
      * intent: str
      * confidence: max softmax prob
      * energy: float
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        inputs.to(device)  # Move inputs to the same device as the model
        logits = model(**inputs).logits[0]

    p   = max_softmax_prob(logits, T)
    e   = energy_score(logits, T)
    pred_id = int(logits.argmax().item())
    intent = model.config.id2label[pred_id]

    if p < tau_prob or e > tau_energy:
        return {"intent": "oos", "confidence": p, "energy": e}

    return {"intent": intent, "confidence": p, "energy": e}
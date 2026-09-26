import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms


class ResizePad:
    def __init__(self, size, fill):
        self.size = int(size)
        self.fill = tuple(fill)

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class FishoraClassifier:
    """Species classifier behind two rejection gates, sharing one backbone pass.

    Gate 1 (fish gate): logistic head on the L2-normalised embedding, rejects
    photos that do not show a fish. Gate 2 (known-species gate): cosine distance
    to the k-th nearest training embedding, rejects fish unlike any training
    photo. Without gate files in the export this behaves like the plain wrapper.
    """

    def __init__(self, export_dir, device=None):
        export_dir = Path(export_dir)
        with open(export_dir / "inference_config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = timm.create_model(self.cfg["model_name"], pretrained=False, num_classes=self.cfg["num_classes"])
        try:
            self.model.reset_classifier(num_classes=self.cfg["num_classes"], global_pool="token")
        except Exception:
            pass
        state = torch.load(export_dir / "model_state_dict.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            ResizePad(self.cfg["img_size"], self.cfg["fill_rgb"]),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg["mean"], self.cfg["std"]),
        ])

        self.gate = self.cfg.get("gate")
        if self.gate:
            head = torch.load(export_dir / self.gate["fish_head_file"], map_location="cpu", weights_only=True)
            self.fish_w = head["weight"].float().to(self.device)
            self.fish_b = head["bias"].float().to(self.device)
            bank = np.load(export_dir / self.gate["knn_bank_file"]).astype(np.float32)
            self.bank = torch.from_numpy(bank).to(self.device)

    @torch.inference_mode()
    def score_batch(self, x):
        """Logits plus gate scores for a preprocessed batch. The suite calls this
        directly so evaluation and serving share one code path."""
        feats = self.model.forward_head(self.model.forward_features(x), pre_logits=True).float()
        logits = self.model.get_classifier()(feats).float()
        if not self.gate:
            return logits, None, None
        z = torch.nn.functional.normalize(feats, dim=1)
        fish_prob = torch.sigmoid(z @ self.fish_w.T + self.fish_b).squeeze(1)
        k = int(self.gate["knn_k"])
        sims = z @ self.bank.T
        knn_dist = 1.0 - sims.topk(k, dim=1).values[:, -1]
        return logits, fish_prob, knn_dist

    @torch.inference_mode()
    def predict(self, image, top_k=3):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits, fish_prob, knn_dist = self.score_batch(x)
        probs = torch.softmax(logits / float(self.cfg["temperature"]), dim=1)[0]

        k = min(top_k, len(self.cfg["classes"]))
        values, indices = probs.topk(k)
        candidates = [
            {"label": self.cfg["classes"][int(idx)], "confidence": float(value)}
            for value, idx in zip(values.cpu(), indices.cpu())
        ]
        best = candidates[0]
        threshold = float(self.cfg["abstain_threshold"])
        status = "confident_prediction" if best["confidence"] >= threshold else "low_confidence_human_verification_required"

        result = {"status": status, "prediction": best, "top_candidates": candidates, "threshold": threshold}
        if self.gate:
            fish_prob, knn_dist = float(fish_prob[0]), float(knn_dist[0])
            if fish_prob < self.gate["fish_threshold"]:
                result["status"] = "rejected_not_fish"
            elif knn_dist > self.gate["knn_threshold"]:
                result["status"] = "rejected_unknown_species"
            result["gate"] = {
                "fish_prob": fish_prob, "fish_threshold": self.gate["fish_threshold"],
                "knn_distance": knn_dist, "knn_threshold": self.gate["knn_threshold"],
            }
        return result

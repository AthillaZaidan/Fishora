import json
from pathlib import Path
import torch
import timm
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
    def __init__(self, export_dir, device=None):
        export_dir = Path(export_dir)
        with open(export_dir / "inference_config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = timm.create_model(
            self.cfg["model_name"],
            pretrained=False,
            num_classes=self.cfg["num_classes"],
        )
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

    @torch.inference_mode()
    def predict(self, image, top_k=3):
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits.float() / float(self.cfg["temperature"]), dim=1)[0]

        k = min(top_k, len(self.cfg["classes"]))
        values, indices = probs.topk(k)
        candidates = [
            {"label": self.cfg["classes"][int(idx)], "confidence": float(value)}
            for value, idx in zip(values.cpu(), indices.cpu())
        ]

        best = candidates[0]
        threshold = float(self.cfg["abstain_threshold"])

        return {
            "status": (
                "confident_prediction"
                if best["confidence"] >= threshold
                else "low_confidence_human_verification_required"
            ),
            "prediction": best,
            "top_candidates": candidates,
            "threshold": threshold,
        }

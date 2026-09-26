from pathlib import Path


class FilesystemImageStore:
    """Saves accepted images under an opaque prediction-reference filename."""

    EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png"}
    CONTENT_TYPES = {extension: content_type for content_type, extension in EXTENSIONS.items()}

    def __init__(self, storage_dir: Path):
        self._storage_dir = Path(storage_dir)

    def save(self, prediction_id: str, image_bytes: bytes, content_type: str) -> str:
        filename = f"{prediction_id}.{self.EXTENSIONS[content_type]}"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        (self._storage_dir / filename).write_bytes(image_bytes)
        return f"images/{filename}"

    def delete(self, image_reference: str) -> None:
        # reference is "images/{filename}"; strip the opaque prefix to stay inside storage_dir
        (self._storage_dir / Path(image_reference).name).unlink(missing_ok=True)

    def find(self, prediction_id: str) -> tuple[Path, str] | None:
        """The stored photo of a prediction and its content type, if one exists.

        Looked up by prediction id rather than by the stored reference, because
        `save` names the file after the id: a lot listing can say whether each
        lot has a photo without a database read per lot. Only names this store
        writes are tried, so an id can never walk out of storage_dir.
        """
        if not prediction_id or Path(prediction_id).name != prediction_id:
            return None
        for extension, content_type in self.CONTENT_TYPES.items():
            path = self._storage_dir / f"{prediction_id}.{extension}"
            if path.is_file():
                return path, content_type
        return None

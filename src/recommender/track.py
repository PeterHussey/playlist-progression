"""Track dataclass — core data unit for the playlist-progression pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    """Represents a single audio track with metadata and extracted features."""

    id: int = 0
    file_path: Path | None = None
    title: str | None = None
    artist: str | None = None
    duration_sec: float = 0.0
    features: list[float] | None = None
    feature_json: str | None = None
    clap_embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if self.file_path is not None and not isinstance(self.file_path, Path):
            object.__setattr__(self, "file_path", Path(self.file_path))

    # ---- Getters ----

    def get_id(self) -> int:
        return self.id

    def get_file_path(self) -> Path | None:
        return self.file_path

    def get_title(self) -> str | None:
        return self.title

    def get_artist(self) -> str | None:
        return self.artist

    def get_duration_sec(self) -> float:
        return self.duration_sec

    def get_features(self) -> list[float] | None:
        return self.features

    def get_feature_json(self) -> str | None:
        return self.feature_json

    def get_clap_embedding(self) -> list[float] | None:
        return self.clap_embedding

    # ---- Setters ----

    def set_id(self, id: int) -> None:
        self.id = id

    def set_file_path(self, file_path: Path | str | None) -> None:
        if file_path is not None:
            file_path = Path(file_path)
        object.__setattr__(self, "file_path", file_path)

    def set_title(self, title: str | None) -> None:
        self.title = title

    def set_artist(self, artist: str | None) -> None:
        self.artist = artist

    def set_duration_sec(self, duration_sec: float) -> None:
        self.duration_sec = duration_sec

    def set_features(self, features: list[float] | None) -> None:
        self.features = features

    def set_feature_json(self, feature_json: str | None) -> None:
        self.feature_json = feature_json

    def set_clap_embedding(self, clap_embedding: list[float] | None) -> None:
        self.clap_embedding = clap_embedding

    # ---- JSON serialisation helpers ----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": str(self.file_path) if self.file_path is not None else None,
            "title": self.title,
            "artist": self.artist,
            "duration_sec": self.duration_sec,
            "features": self.features,
            "feature_json": self.feature_json,
            "clap_embedding": self.clap_embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Track:
        t = cls(
            id=d.get("id", 0),
            file_path=d.get("file_path"),
            title=d.get("title"),
            artist=d.get("artist"),
            duration_sec=d.get("duration_sec", 0.0),
            features=d.get("features"),
            feature_json=d.get("feature_json"),
            clap_embedding=d.get("clap_embedding"),
        )
        return t
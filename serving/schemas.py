from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class MetricsResponse(BaseModel):
    model: str
    miou: float
    classes: int


class PredictionResponse(BaseModel):
    original_image: str
    result_image: str

    crack: float
    spall: float
    corrosion: float
    efflorescence: float

    latency_ms: int
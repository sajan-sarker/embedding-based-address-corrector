from pydantic import BaseModel, Field


class CorrectRequest(BaseModel):
    address: str = Field(..., min_length=1, max_length=2000, description="Raw (possibly misspelled) Bangla address")
    n_results: int | None = Field(None, ge=1, le=100, description="Vector search candidates per collection")
    padding: int | None = Field(None, ge=0, le=20, description="Extra characters in the n-gram sliding window")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "address": "রহমান মঞ্জিল, বাড়ি নং: ৩৪, রোড নং: ৬, সেক্টর: ১০, উত্তর মডেল টাউন, উতরা, ডাকঘর: উতর - ১২৩০, উতরা পশ্চম, ঢাকা",
                "n_results": 15,
                "padding": 0,
            }]
        }
    }


class IngestRequest(BaseModel):
    force: bool = Field(True, description="Re-ingest even when collections are already up to date")
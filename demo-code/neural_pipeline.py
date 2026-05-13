"""
A-Eyes Neural Processing Pipeline
Real-time screen analysis engine with multi-modal fusion
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Callable, AsyncGenerator
from enum import Enum
from collections import deque
import time


# ══════════════════════════════════════════════════════════════
# Configuration & Types
# ══════════════════════════════════════════════════════════════

class AnalysisMode(Enum):
    VISUAL   = "visual"
    AUDIO    = "audio"
    COMBINED = "combined"
    DEEP     = "deep_inspection"

class Priority(Enum):
    LOW      = 0
    NORMAL   = 1
    HIGH     = 2
    CRITICAL = 3

@dataclass
class FrameSignature:
    """Perceptual hash of a captured frame for change detection."""
    timestamp: float
    hash_value: str
    delta_score: float = 0.0
    region_hashes: dict = field(default_factory=dict)

    def divergence_from(self, other: 'FrameSignature') -> float:
        if not other:
            return 1.0
        matching = sum(a == b for a, b in zip(self.hash_value, other.hash_value))
        return 1.0 - (matching / max(len(self.hash_value), 1))


@dataclass
class AnalysisResult:
    """Container for multi-modal analysis output."""
    frame_id: str
    mode: AnalysisMode
    confidence: float
    annotations: list[dict]
    transcript_segment: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# Pipeline Decorators
# ══════════════════════════════════════════════════════════════

def pipeline_stage(name: str, retries: int = 3):
    """Decorator that wraps async pipeline stages with retry logic."""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    start = time.perf_counter()
                    result = await func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"  ✓ [{name}] completed in {elapsed:.1f}ms")
                    return result
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"  ✗ [{name}] failed after {retries} attempts: {e}")
                        raise
                    await asyncio.sleep(0.1 * (attempt + 1))
            return None
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════
# Core Processing Engine
# ══════════════════════════════════════════════════════════════

class NeuralPipeline:
    """
    Multi-stage async pipeline for real-time screen analysis.
    Implements adaptive frame selection, perceptual hashing,
    and parallel multi-modal processing.
    """

    CHANGE_THRESHOLD = 0.035
    BURST_WINDOW_MS = 800
    MAX_QUEUE_DEPTH = 256
    HASH_DIMENSIONS = (16, 16)

    def __init__(
        self,
        mode: AnalysisMode = AnalysisMode.COMBINED,
        max_concurrent: int = 4,
        enable_audio: bool = True,
    ):
        self.mode = mode
        self.max_concurrent = max_concurrent
        self.enable_audio = enable_audio
        self._frame_buffer: deque[FrameSignature] = deque(maxlen=self.MAX_QUEUE_DEPTH)
        self._processing_semaphore = asyncio.Semaphore(max_concurrent)
        self._results_cache: dict[str, AnalysisResult] = {}
        self._baseline: Optional[FrameSignature] = None
        self._burst_active = False
        self._stats = {"frames_processed": 0, "bursts_triggered": 0, "cache_hits": 0}

    # ── Frame Ingestion ──────────────────────────────────────

    @pipeline_stage("frame_ingest")
    async def ingest_frame(self, raw_pixels: bytes, timestamp: float) -> FrameSignature:
        """Compute perceptual hash and detect significant changes."""
        hash_value = hashlib.sha256(raw_pixels).hexdigest()[:32]

        signature = FrameSignature(
            timestamp=timestamp,
            hash_value=hash_value,
            region_hashes=self._compute_region_hashes(raw_pixels),
        )

        if self._baseline:
            signature.delta_score = signature.divergence_from(self._baseline)

        self._frame_buffer.append(signature)
        return signature

    def _compute_region_hashes(self, raw_pixels: bytes) -> dict:
        """Split frame into grid regions and hash each independently."""
        chunk_size = max(len(raw_pixels) // 16, 1)
        regions = {}
        for i in range(16):
            start = i * chunk_size
            chunk = raw_pixels[start:start + chunk_size]
            regions[f"r{i:02d}"] = hashlib.md5(chunk).hexdigest()[:8]
        return regions

    # ── Change Detection ─────────────────────────────────────

    @pipeline_stage("change_detect")
    async def detect_changes(self, signature: FrameSignature) -> bool:
        """Determine if frame represents significant visual change."""
        if not self._baseline:
            self._baseline = signature
            return True

        if signature.delta_score >= self.CHANGE_THRESHOLD:
            self._baseline = signature
            if not self._burst_active:
                self._burst_active = True
                self._stats["bursts_triggered"] += 1
                asyncio.create_task(self._burst_cooldown())
            return True

        return False

    async def _burst_cooldown(self):
        await asyncio.sleep(self.BURST_WINDOW_MS / 1000)
        self._burst_active = False

    # ── Multi-Modal Analysis ─────────────────────────────────

    @pipeline_stage("analysis")
    async def analyze_frame(
        self,
        signature: FrameSignature,
        audio_segment: Optional[bytes] = None,
    ) -> AnalysisResult:
        """Run parallel analysis across all configured modalities."""

        frame_id = f"f_{signature.timestamp:.0f}_{signature.hash_value[:8]}"

        # Check cache first
        if frame_id in self._results_cache:
            self._stats["cache_hits"] += 1
            return self._results_cache[frame_id]

        async with self._processing_semaphore:
            tasks = [self._visual_analysis(signature)]

            if self.enable_audio and audio_segment:
                tasks.append(self._audio_analysis(audio_segment))

            # ⚠️ BUG: results is a list of completed tasks, but we
            # incorrectly index into it assuming both tasks always exist
            results = await asyncio.gather(*tasks, return_exceptions=True)

            visual = results[0]
            # This line crashes when audio is disabled — index out of range!
            audio_annotations = results[1].get("segments", [])

            combined = AnalysisResult(
                frame_id=frame_id,
                mode=self.mode,
                confidence=visual.get("confidence", 0.0),
                annotations=visual.get("annotations", []) + audio_annotations,
                transcript_segment=results[1].get("transcript") if len(results) > 1 else None,
                processing_time_ms=visual.get("elapsed_ms", 0),
            )

            self._results_cache[frame_id] = combined
            self._stats["frames_processed"] += 1
            return combined

    async def _visual_analysis(self, sig: FrameSignature) -> dict:
        """Simulate visual feature extraction."""
        await asyncio.sleep(0.05)  # Simulated processing time
        return {
            "confidence": min(0.95, 0.6 + sig.delta_score),
            "annotations": [
                {"type": "region_change", "score": sig.delta_score, "regions": list(sig.region_hashes.keys())},
                {"type": "ui_element", "detected": ["code_editor", "terminal", "error_dialog"]},
            ],
            "elapsed_ms": 50.0,
        }

    async def _audio_analysis(self, audio: bytes) -> dict:
        """Simulate audio transcription + sentiment analysis."""
        await asyncio.sleep(0.08)
        return {
            "transcript": "analyzing the error output on screen",
            "segments": [{"text": "analyzing the error output", "start": 0.0, "end": 2.1}],
            "sentiment": "focused",
        }

    # ── Pipeline Orchestrator ────────────────────────────────

    async def process_stream(
        self, frame_source: AsyncGenerator[tuple[bytes, float], None]
    ) -> list[AnalysisResult]:
        """Main entry point: consume frame stream and produce analysis results."""
        results = []

        print(f"\n{'═' * 60}")
        print(f"  A-Eyes Neural Pipeline v2.0")
        print(f"  Mode: {self.mode.value} | Concurrency: {self.max_concurrent}")
        print(f"  Audio: {'enabled' if self.enable_audio else 'disabled'}")
        print(f"{'═' * 60}\n")

        async for raw_pixels, timestamp in frame_source:
            signature = await self.ingest_frame(raw_pixels, timestamp)
            is_significant = await self.detect_changes(signature)

            if is_significant:
                result = await self.analyze_frame(signature, audio_segment=None)
                results.append(result)
                print(f"    → Frame {result.frame_id}: confidence={result.confidence:.2f}")

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[AnalysisResult]):
        print(f"\n{'─' * 60}")
        print(f"  Pipeline Summary:")
        print(f"    Frames processed : {self._stats['frames_processed']}")
        print(f"    Bursts triggered : {self._stats['bursts_triggered']}")
        print(f"    Cache hits       : {self._stats['cache_hits']}")
        print(f"    Total results    : {len(results)}")
        print(f"{'─' * 60}\n")


# ══════════════════════════════════════════════════════════════
# Demo: Run the Pipeline
# ══════════════════════════════════════════════════════════════

async def simulated_capture() -> AsyncGenerator[tuple[bytes, float], None]:
    """Simulate a screen capture stream with varying content."""
    import os
    base_time = time.time()
    for i in range(12):
        # Generate pseudo-random frame data that changes over time
        noise = os.urandom(1024) + bytes([i % 256]) * 512
        yield (noise, time.time() - base_time)
        await asyncio.sleep(0.15)


async def main():
    pipeline = NeuralPipeline(
        mode=AnalysisMode.COMBINED,
        max_concurrent=4,
        enable_audio=True,   # Audio is enabled, but no audio data is passed
    )

    source = simulated_capture()
    results = await pipeline.process_stream(source)

    print(json.dumps(
        [{"id": r.frame_id, "confidence": r.confidence, "annotations": len(r.annotations)} for r in results],
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())

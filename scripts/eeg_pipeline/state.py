"""Atomic Gate state for interruption-safe EEG workflow execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import os

from .provenance import sha256_file


@dataclass(frozen=True)
class EvidenceRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass
class GateRecord:
    status: str = "pending"
    user_decision: str | None = None
    completed_at_utc: str | None = None
    completed_at_local: str | None = None
    evidence: list[EvidenceRecord] = field(default_factory=list)


@dataclass
class PipelineState:
    state_path: Path
    run_id: str
    gate_order: tuple[str, ...]
    gates: dict[str, GateRecord]

    @classmethod
    def create(
        cls,
        state_path: Path,
        run_id: str,
        gate_order: tuple[str, ...],
    ) -> "PipelineState":
        if not run_id.strip():
            raise ValueError("run_id不能为空")
        if not gate_order or len(gate_order) != len(set(gate_order)):
            raise ValueError("gate_order必须非空且不得重复")
        state = cls(
            state_path=state_path.resolve(),
            run_id=run_id,
            gate_order=gate_order,
            gates={gate: GateRecord() for gate in gate_order},
        )
        state._save_atomic()
        return state

    @classmethod
    def load(cls, state_path: Path) -> "PipelineState":
        resolved = state_path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        gates = {
            gate: GateRecord(
                status=record["status"],
                user_decision=record.get("user_decision"),
                completed_at_utc=record.get("completed_at_utc"),
                completed_at_local=record.get("completed_at_local"),
                evidence=[EvidenceRecord(**item) for item in record["evidence"]],
            )
            for gate, record in payload["gates"].items()
        }
        return cls(
            state_path=resolved,
            run_id=payload["run_id"],
            gate_order=tuple(payload["gate_order"]),
            gates=gates,
        )

    @property
    def next_incomplete_gate(self) -> str | None:
        for gate in self.gate_order:
            if self.gates[gate].status != "complete":
                return gate
        return None

    def complete_gate(
        self,
        gate: str,
        evidence: list[Path],
        user_decision: str | None = None,
    ) -> None:
        if gate not in self.gates:
            raise KeyError(f"未知Gate：{gate}")
        if self.gates[gate].status == "complete":
            raise RuntimeError(f"Gate已经完成：{gate}")

        records: list[EvidenceRecord] = []
        for path in evidence:
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            records.append(
                EvidenceRecord(
                    path=str(resolved),
                    size_bytes=resolved.stat().st_size,
                    sha256=sha256_file(resolved),
                )
            )

        utc_now = datetime.now(timezone.utc)
        local_now = utc_now.astimezone()
        self.gates[gate] = GateRecord(
            status="complete",
            user_decision=user_decision,
            completed_at_utc=utc_now.isoformat(),
            completed_at_local=local_now.isoformat(),
            evidence=records,
        )
        self._save_atomic()

    def _save_atomic(self) -> None:
        payload = {
            "run_id": self.run_id,
            "gate_order": list(self.gate_order),
            "gates": {
                gate: asdict(self.gates[gate]) for gate in self.gate_order
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)

"""
Sentry Flyer — Mission Reporting & SitRep Compiler
Location: Phase 4/report_generator.py
Purpose: Reads the mission database and exports (a) a GeoJSON FeatureCollection
         for GIS tools and (b) a compact LoRa text packet for low-bandwidth
         relay.

LoRa packet format
------------------
    $SF,<count>,<id>:<class>:<lat>:<lon>:<sev>-<id>:...*<XX>

* Targets are ordered by priority (severity x confidence) so if the packet
  must be truncated to fit the radio's payload limit, the least important
  targets are dropped first. <count> reflects what is actually in the packet.
* <XX> is an NMEA-style XOR checksum of everything between '$' and '*'.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("sentry.report")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "Phase 1" / "sentry_mission.db"

# LoRa (SX127x) hard payload limit is 255 bytes; leave headroom for framing.
LORA_MAX_PAYLOAD_BYTES = 240

CLASS_CODES = {
    "person": "P",
    "person_unverified": "p",
    "fire": "F",
    "thermal_anomaly": "f",
    "smoke": "S",
    "floodwater": "W",
    "damage": "D",
    "environmental_anomaly": "E",
    "reflector_false_alarm": "R",
}


def nmea_checksum(body: str) -> str:
    checksum = 0
    for ch in body:
        checksum ^= ord(ch)
    return f"{checksum:02X}"


def verify_lora_packet(packet: str) -> bool:
    """True if `packet` is well-formed and its checksum matches."""
    if not packet.startswith("$") or "*" not in packet:
        return False
    body, _, given = packet[1:].rpartition("*")
    return nmea_checksum(body) == given.upper()


class MissionReporter:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Mission database not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _prioritised_clusters(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute(
            """
            SELECT *, (max_severity * mean_confidence) AS priority_score
            FROM fused_clusters
            ORDER BY priority_score DESC, last_updated DESC
            """
        ).fetchall()

    # ------------------------------------------------------------------ #
    def build_geojson(self) -> dict:
        """GeoJSON FeatureCollection of all clusters, highest priority first."""
        with self._connection() as conn:
            rows = self._prioritised_clusters(conn)

        features = [
            {
                "type": "Feature",
                "geometry": {
                    # GeoJSON order is [lon, lat].
                    "type": "Point",
                    "coordinates": [row["center_longitude"], row["center_latitude"]],
                },
                "properties": {
                    "id": row["id"],
                    "class_name": row["class_name"],
                    "confidence": round(row["mean_confidence"], 2),
                    "severity_score": row["max_severity"],
                    "priority_score": round(row["priority_score"], 1),
                    "detections_count": row["detection_count"],
                    "first_seen_epoch": row["first_seen"] if "first_seen" in row.keys() else None,
                    "last_seen_epoch": row["last_updated"],
                },
            }
            for row in rows
        ]
        return {"type": "FeatureCollection", "features": features}

    def export_geojson_report(self, output_path: str | Path = "mission_report.geojson") -> bool:
        try:
            data = self.build_geojson()
        except (FileNotFoundError, sqlite3.OperationalError) as e:
            log.error("Could not build GeoJSON report: %s", e)
            return False
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("GeoJSON report with %d targets written to %s", len(data["features"]), output_path)
        return True

    # ------------------------------------------------------------------ #
    def build_lora_packet(self, max_bytes: int = LORA_MAX_PAYLOAD_BYTES) -> str:
        """Dense ASCII packet, truncated by priority to fit `max_bytes`."""
        with self._connection() as conn:
            rows = self._prioritised_clusters(conn)

        segments = [
            f"{row['id']}:{CLASS_CODES.get(row['class_name'], 'U')}:"
            f"{row['center_latitude']:.5f}:{row['center_longitude']:.5f}:{row['max_severity']}"
            for row in rows
        ]

        def assemble(segs: list[str]) -> str:
            body = f"SF,{len(segs)}," + ("-".join(segs) if segs else "EMPTY")
            return f"${body}*{nmea_checksum(body)}"

        packet = assemble(segments)
        while len(packet.encode("ascii")) > max_bytes and segments:
            segments.pop()  # lowest priority goes first
            packet = assemble(segments)

        if len(segments) < len(rows):
            log.warning("LoRa packet truncated: %d of %d targets fit in %d bytes", len(segments), len(rows), max_bytes)
        return packet

    def generate_micro_lora_packet(self, output_path: str | Path = "lora_packet.txt") -> Optional[str]:
        try:
            packet = self.build_lora_packet()
        except (FileNotFoundError, sqlite3.OperationalError) as e:
            log.error("Could not build LoRa packet: %s", e)
            return None
        Path(output_path).write_text(packet, encoding="ascii")
        log.info("LoRa packet of %d bytes written to %s", len(packet), output_path)
        return packet


# =====================================================================
# SYSTEM INTEGRATION TEST
# =====================================================================
if __name__ == "__main__":
    import os
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("[TEST] Initializing Sentry Flyer Mission Reporter...")

    tmp_dir = Path(tempfile.mkdtemp())
    db_file = tmp_dir / "sentry_mission_temp.db"
    schema = Path(__file__).resolve().parent.parent / "Phase 1" / "schema.sql"

    conn = sqlite3.connect(db_file)
    conn.executescript(schema.read_text(encoding="utf-8"))
    conn.executemany(
        """
        INSERT INTO fused_clusters
            (class_name, center_latitude, center_longitude, mean_confidence, max_severity,
             detection_count, first_seen, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("person", 25.31025, 78.48810, 0.94, 98, 4, 1711111100.0, 1711111111.0),
            ("fire", 25.31140, 78.48930, 0.89, 85, 2, 1711111110.0, 1711111115.0),
        ],
    )
    conn.commit()
    conn.close()

    reporter = MissionReporter(db_path=db_file)
    reporter.export_geojson_report(output_path=tmp_dir / "test_mission_report.geojson")
    lora_data = reporter.generate_micro_lora_packet(output_path=tmp_dir / "test_lora_packet.txt")

    print("\n=== GENERATED LORA PACKET PAYLOAD ===")
    print(lora_data)
    print(f"Total String Size: {len(lora_data)} bytes | Checksum OK: {verify_lora_packet(lora_data)}")
    print("======================================\n")

    for f in tmp_dir.iterdir():
        os.remove(f)
    os.rmdir(tmp_dir)

#!/usr/bin/env python3
# ==============================================================================
# gaudi_topology.py
# Provides GaudiTopology class:
#   - discover all Gaudi cards via hl-smi
#   - return NUMA node and CPU IDs per card
# Works with hl-smi v1.22.0+ (HL-325L / Gaudi3) table format.
# ==============================================================================

import subprocess
import re
import os
from typing import List, Dict, Optional
import shutil

class GaudiTopology:
    """Utility class to discover Gaudi cards and their NUMA / CPU locality."""

    def __init__(self):
        self.cards = self._discover_cards()

    # ------------------------------------------------------------------
    def _run_cmd(self, cmd: str) -> str:
        """Run a shell command and return stdout."""
        try:
            result = subprocess.run(cmd, shell=True, check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {cmd}\n{e.stderr}")

    # ------------------------------------------------------------------
    def _parse_hl_smi_table(self, text: str) -> List[Dict]:
        """
        Parse hl-smi v1.22+ table format.
        Example line:
        |   0  HL-325L             N/A  | 0000:97:00.0     N/A | ...
        """
        cards = []
        pattern = re.compile(
            r'^\|\s*(\d+)\s+([A-Z0-9-]+)\s+N/A\s+\|\s*([0-9a-fA-F:.]+)\s+N/A\s*\|'
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            card_id, model, bus_id = match.groups()
            if not bus_id.startswith("0000:"):
                bus_id = "0000:" + bus_id
            cards.append({
                "card_id": int(card_id),
                "model": model,
                "bus_id": bus_id
            })
        return cards

    # ------------------------------------------------------------------
    def _get_sysfs_info(self, bus_id: str) -> Dict[str, Optional[str]]:
        """Fetch NUMA node and local CPU list from sysfs."""
        sys_path = f"/sys/bus/pci/devices/{bus_id}"
        info = {"numa_node": None, "local_cpulist": None}
        try:
            with open(os.path.join(sys_path, "numa_node")) as f:
                info["numa_node"] = f.read().strip()
        except FileNotFoundError:
            pass
        try:
            with open(os.path.join(sys_path, "local_cpulist")) as f:
                info["local_cpulist"] = f.read().strip()
        except FileNotFoundError:
            pass
        return info

    # ------------------------------------------------------------------
    def _discover_cards(self) -> List[Dict]:
        """Run hl-smi and discover Gaudi cards."""
        if shutil.which("hl-smi") is None:
            print("No hl-smi found")
            return None

        hl_smi_output = self._run_cmd("hl-smi")
        cards = self._parse_hl_smi_table(hl_smi_output)
        for c in cards:
            sysfs_info = self._get_sysfs_info(c["bus_id"])
            c.update(sysfs_info)
        return cards

    # ------------------------------------------------------------------
    def get_cards(self) -> List[Dict]:
        """Return list of all discovered cards sorted by NUMA node (then card_id)."""
        def sort_key(c):
            # Convert numa_node to int when possible, else put N/A at the end
            try:
                return (int(c["numa_node"]), c["card_id"])
            except (TypeError, ValueError):
                return (999, c["card_id"])
        return sorted(self.cards, key=sort_key)


    # ------------------------------------------------------------------
    def get_numa_for_card(self, card_id: int) -> Optional[str]:
        """Return NUMA node for a given card ID."""
        for c in self.cards:
            if c["card_id"] == card_id:
                return c["numa_node"]
        return None

    # ------------------------------------------------------------------
    def get_cpus_for_card(self, card_id: int) -> Optional[str]:
        """Return local CPU list for a given card ID."""
        for c in self.cards:
            if c["card_id"] == card_id:
                return c["local_cpulist"]
        return None

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    topo = GaudiTopology()
    for card in topo.get_cards():
        print(f"Card {card['card_id']} ({card['model']}):")
        print(f"  Bus ID     : {card['bus_id']}")
        print(f"  NUMA Node  : {card['numa_node']}")
        print(f"  Local CPUs : {card['local_cpulist']}")
        print()


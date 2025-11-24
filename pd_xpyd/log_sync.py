#!/usr/bin/env python3
"""
Stage 2: Sync log files from local disk to NFS.
This runs as a background process, copying local log files to NFS periodically.
"""

import argparse
import os
import shutil
import signal
import sys
import time
from pathlib import Path


class LogSyncer:
    def __init__(
        self,
        local_dir: str,
        nfs_dir: str,
        sync_interval: float = 2.0,
        follow: bool = True,
        max_size: int = 10 * 1024 * 1024,  # 10MB default
    ):
        self.local_dir = Path(local_dir)
        self.nfs_dir = Path(nfs_dir)
        self.sync_interval = sync_interval
        self.follow = follow
        self.max_size = max_size
        self.running = True
        
        # Ensure directories exist
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.nfs_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last sync positions for each file
        self.last_positions = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals to sync and exit gracefully."""
        self.running = False

    def _rotate_file(self, nfs_path: Path):
        """Rotate NFS log file when it exceeds max_size.
        Rotates to .1, .2, etc. and removes oldest if too many exist.
        """
        if not nfs_path.exists():
            return
        
        # Find existing rotated files (e.g., file.log.1, file.log.2, ...)
        rotated_files = []
        parent = nfs_path.parent
        base_name = nfs_path.name
        
        for f in parent.glob(f"{base_name}.*"):
            try:
                # Extract rotation number
                suffix = f.name[len(base_name) + 1:]  # Everything after the dot
                if suffix.isdigit():
                    rotated_files.append((int(suffix), f))
            except (ValueError, IndexError):
                continue
        
        # Sort by rotation number
        rotated_files.sort(key=lambda x: x[0])
        
        # Rotate existing files: .N -> .(N+1)
        for rot_num, rot_file in reversed(rotated_files):
            new_name = parent / f"{base_name}.{rot_num + 1}"
            rot_file.rename(new_name)
        
        # Rotate current file to .1
        rotated_path = parent / f"{base_name}.1"
        nfs_path.rename(rotated_path)
        
        # Note: We don't reset last_positions here because we're rotating the NFS file,
        # not the local file. We continue syncing from where we left off in the local file.

    def sync_file(self, local_path: Path, nfs_path: Path):
        """Sync a single file from local to NFS, appending new content."""
        try:
            if not local_path.exists():
                return
            
            # Get current file size
            current_size = local_path.stat().st_size
            
            # Get last known position for this file
            last_pos = self.last_positions.get(str(local_path), 0)
            
            # If file shrunk (rotation/recreation), reset position
            if current_size < last_pos:
                last_pos = 0
            
            # If file hasn't grown, nothing to sync
            if current_size <= last_pos:
                return
            
            # Read new content
            with open(local_path, "rb") as f:
                f.seek(last_pos)
                new_content = f.read()
            
            if new_content:
                # Check if NFS file needs rotation before appending
                nfs_path.parent.mkdir(parents=True, exist_ok=True)
                if nfs_path.exists():
                    nfs_size = nfs_path.stat().st_size
                    if nfs_size >= self.max_size:
                        self._rotate_file(nfs_path)
                
                # Append to NFS file
                with open(nfs_path, "ab") as f:
                    f.write(new_content)
                    f.flush()
                    # Sync to ensure data is written to NFS
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass  # fsync may fail on some file systems
                
                # Update last position
                self.last_positions[str(local_path)] = current_size
        
        except Exception as e:
            print(f"ERROR syncing {local_path} to {nfs_path}: {e}", file=sys.stderr)

    def sync_all(self):
        """Sync all log files from local directory to NFS directory."""
        if not self.local_dir.exists():
            return
        
        # Find all log files in local directory
        for local_file in self.local_dir.glob("*.log"):
            # Maintain same directory structure in NFS
            relative_path = local_file.relative_to(self.local_dir)
            nfs_file = self.nfs_dir / relative_path
            
            self.sync_file(local_file, nfs_file)

    def run(self):
        """Main sync loop."""
        max_size_mb = self.max_size / (1024 * 1024)
        print(f"LogSyncer: Syncing {self.local_dir} -> {self.nfs_dir} every {self.sync_interval}s (max size: {max_size_mb:.1f}MB)", file=sys.stderr)
        
        while self.running:
            try:
                self.sync_all()
            except Exception as e:
                print(f"ERROR in sync loop: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
            
            if self.follow and self.running:
                time.sleep(self.sync_interval)
            else:
                break
        
        # Final sync before exit
        self.sync_all()
        print("LogSyncer: Exiting", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Sync log files from local disk to NFS")
    parser.add_argument("local_dir", help="Local directory containing log files")
    parser.add_argument("nfs_dir", help="NFS directory to sync to")
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=2.0,
        help="Sync interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Sync once and exit (default: continuous sync)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum NFS log file size in bytes before rotation (default: 10MB)",
    )

    args = parser.parse_args()

    syncer = LogSyncer(
        local_dir=args.local_dir,
        nfs_dir=args.nfs_dir,
        sync_interval=args.sync_interval,
        follow=not args.once,
        max_size=args.max_size,
    )

    syncer.run()


if __name__ == "__main__":
    main()



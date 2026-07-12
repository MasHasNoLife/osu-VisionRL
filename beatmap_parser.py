"""
Beatmap Parser for osu! .osu files.
Extracts hit object positions and timings for dense reward shaping.
"""

import os
from typing import Optional, Tuple, List


class BeatmapParser:
    """Parses osu! beatmap files to extract hit object data."""
    
    OSU_PLAYFIELD_W = 512  # osu! coordinate system width
    OSU_PLAYFIELD_H = 384  # osu! coordinate system height
    
    def __init__(self, songs_dir: str):
        self.songs_dir = songs_dir
        self.objects: List[dict] = []
        self.pointer = 0  # Index of next object to aim for
        self.cs = 4.0  # Circle Size (default)
        self.ar = 5.0  # Approach Rate (default)
        self.loaded_map = ""
    
    def load_beatmap(self, folder: str, filename: str) -> bool:
        """Load and parse a .osu beatmap file."""
        filepath = os.path.join(self.songs_dir, folder, filename)
        
        if not os.path.exists(filepath):
            print(f"[BeatmapParser] File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[BeatmapParser] Error reading file: {e}")
            return False
        
        self.objects = []
        self.pointer = 0
        
        # Parse sections
        current_section = ""
        for line in lines:
            line = line.strip()
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
            
            if not line or line.startswith('//'):
                continue
            
            # Parse difficulty settings
            if current_section == "Difficulty":
                if line.startswith("CircleSize:"):
                    self.cs = float(line.split(":")[1])
                elif line.startswith("ApproachRate:"):
                    self.ar = float(line.split(":")[1])
            
            # Parse hit objects
            if current_section == "HitObjects":
                self._parse_hit_object(line)
        
        # Sort by time
        self.objects.sort(key=lambda o: o['time'])
        
        self.loaded_map = f"{folder}/{filename}"
        print(f"[BeatmapParser] Loaded {len(self.objects)} hit objects from {filename}")
        print(f"[BeatmapParser] CS={self.cs}, AR={self.ar}")
        
        return True
    
    def _parse_hit_object(self, line: str):
        """Parse a single hit object line."""
        try:
            parts = line.split(',')
            if len(parts) < 4:
                return
            
            x = int(parts[0])
            y = int(parts[1])
            time_ms = int(parts[2])
            obj_type = int(parts[3])
            
            # Decode type bitmask
            if obj_type & 8:
                # Spinner — always at center, not useful for aiming
                return
            elif obj_type & 2:
                # Slider — we care about the start position
                self.objects.append({
                    'x': x, 'y': y, 'time': time_ms, 'type': 'slider'
                })
            elif obj_type & 1:
                # Hit circle
                self.objects.append({
                    'x': x, 'y': y, 'time': time_ms, 'type': 'circle'
                })
        except (ValueError, IndexError):
            pass  # Skip malformed lines
    
    def get_upcoming(self, current_time_ms: int, n: int = 2) -> List[dict]:
        """
        Get up to n upcoming hit objects, starting with the current aim target.

        Advances the pointer past objects whose timing window has expired.
        An object is considered expired if current_time > object_time + 200ms
        (the widest 50-hit window is ~±200ms at OD0, so past that the object
        is gone and the next one is the aiming target).
        """
        while (self.pointer < len(self.objects) and
               current_time_ms > self.objects[self.pointer]['time'] + 200):
            self.pointer += 1

        return self.objects[self.pointer:self.pointer + n]

    def get_next_object(self, current_time_ms: int) -> Optional[dict]:
        """Get the next hit object the AI should aim for (None if map is done)."""
        upcoming = self.get_upcoming(current_time_ms, 1)
        return upcoming[0] if upcoming else None
    
    def get_screen_coords(self, osu_x: float, osu_y: float,
                          play_left: int, play_right: int,
                          play_top: int, play_bottom: int) -> Tuple[int, int]:
        """
        Convert osu! playfield coordinates (0-512, 0-384) to screen pixel coordinates.
        """
        screen_x = play_left + (osu_x / self.OSU_PLAYFIELD_W) * (play_right - play_left)
        screen_y = play_top + (osu_y / self.OSU_PLAYFIELD_H) * (play_bottom - play_top)
        return int(screen_x), int(screen_y)
    
    def reset(self):
        """Reset the pointer for a new episode/retry."""
        self.pointer = 0
    
    @property
    def num_objects(self) -> int:
        """Total number of parsed hit objects."""
        return len(self.objects)


if __name__ == "__main__":
    songs_dir = "/home/mas/.osu-wine/drive_c/users/mas/AppData/Local/osu!/Songs"
    parser = BeatmapParser(songs_dir)
    
    # Load the training map
    folder = "2239337"
    filename = "Junichi Masuda - Sentou! Champion Iris (Sotarks) [Noffy's Easy].osu"
    
    if parser.load_beatmap(folder, filename):
        print(f"\nTotal objects: {parser.num_objects}")
        print(f"\nFirst 10 objects:")
        for i, obj in enumerate(parser.objects[:10]):
            print(f"  [{i}] ({obj['x']:3d}, {obj['y']:3d}) at {obj['time']}ms  [{obj['type']}]")
        
        # Test get_next_object
        print(f"\nTesting get_next_object():")
        for test_time in [0, 10000, 20000, 50000]:
            parser.reset()
            obj = parser.get_next_object(test_time)
            if obj:
                print(f"  At {test_time}ms -> next: ({obj['x']}, {obj['y']}) at {obj['time']}ms")
            else:
                print(f"  At {test_time}ms -> no more objects")
        
        # Test coordinate conversion
        print(f"\nTesting get_screen_coords() (play bounds: L=370, R=2190, T=72, B=1368):")
        for obj in parser.objects[:5]:
            sx, sy = parser.get_screen_coords(obj['x'], obj['y'], 370, 2190, 72, 1368)
            print(f"  osu({obj['x']}, {obj['y']}) -> screen({sx}, {sy})")

"""Simple loitering detector utility.

Tracks per-object positions and timestamps (frame indices) and flags loitering
when an object remains within a small displacement for a configured time.

API:
  detector = LoiterDetector(loiter_seconds=10, min_disp_pixels=20, fps=30)
  is_loitering = detector.update(track_id, bbox, frame_idx)
  detector.handle_lost(track_id, frame_idx)
"""
from collections import defaultdict
import math


class LoiterDetector:
    def __init__(self, loiter_seconds=10.0, min_disp_pixels=20.0, max_gap_seconds=1.0, fps=30):
        # threshold: how many seconds of near-stationary to call loitering
        self.loiter_seconds = float(loiter_seconds)
        # movement threshold in pixels (if displacement between updates is below this, count as stationary)
        self.min_disp_pixels = float(min_disp_pixels)
        # allow short gaps in tracking (occlusion)
        self.max_gap_seconds = float(max_gap_seconds)
        self.fps = float(fps)

        # state keyed by track_id
        self.state = defaultdict(lambda: {
            'first_stationary_frame': None,
            'last_frame': None,
            'last_pos': None,
            'gap_start_frame': None,
            'alerted': False,
        })

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, track_id, bbox, frame_idx):
        """Update the track state for a given frame.

        Args:
            track_id: integer track id
            bbox: (x1, y1, x2, y2) in pixels
            frame_idx: integer frame index (monotonic increasing)

        Returns:
            True if loitering is detected for this track at this frame, else False.
        """
        st = self.state[track_id]
        cx, cy = self._center(bbox)

        # initialize
        if st['last_frame'] is None:
            st['last_frame'] = frame_idx
            st['last_pos'] = (cx, cy)
            st['first_stationary_frame'] = frame_idx
            st['gap_start_frame'] = None
            st['alerted'] = False
            return False

        # compute displacement since last seen
        dx = cx - st['last_pos'][0]
        dy = cy - st['last_pos'][1]
        disp = math.hypot(dx, dy)

        # reset gap if present
        if st['gap_start_frame'] is not None:
            # if gap is small, keep state
            gap_seconds = (frame_idx - st['gap_start_frame']) / self.fps
            if gap_seconds > self.max_gap_seconds:
                # gap too large -> reset
                st['first_stationary_frame'] = frame_idx
                st['last_pos'] = (cx, cy)
                st['last_frame'] = frame_idx
                st['gap_start_frame'] = None
                st['alerted'] = False
                return False
            else:
                # still within allowed gap, treat as continuous
                st['gap_start_frame'] = None

        # update last seen
        st['last_frame'] = frame_idx
        st['last_pos'] = (cx, cy)

        # if displacement below threshold, extend stationary period; else reset stationary start
        if disp <= self.min_disp_pixels:
            # remain/enter stationary
            if st['first_stationary_frame'] is None:
                st['first_stationary_frame'] = frame_idx
        else:
            # moving - reset stationary timer
            st['first_stationary_frame'] = frame_idx
            st['alerted'] = False
            return False

        # compute stationary duration in seconds
        duration_seconds = (frame_idx - st['first_stationary_frame']) / self.fps

        if duration_seconds >= self.loiter_seconds and not st['alerted']:
            st['alerted'] = True
            return True

        return False

    def handle_lost(self, track_id, frame_idx):
        """Call when a track is lost for a frame; starts gap timer to allow brief occlusions."""
        st = self.state.get(track_id)
        if st is None:
            return
        if st['gap_start_frame'] is None:
            st['gap_start_frame'] = frame_idx

    def reset(self):
        self.state.clear()

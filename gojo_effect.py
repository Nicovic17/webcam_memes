"""Relaxed two-finger trigger and a procedural, live-composited Unlimited Void.

No anime footage or static background: clouds, lensing, stars, accretion disk,
barrier expansion and collapse are rendered locally from elapsed time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import mediapipe as mp
import numpy as np

from gojo_comic import GojoComicOverlay
from gojo_notifications import GojoNotifications


HOLD_SECONDS = 0.40
TRACKING_GRACE_SECONDS = 0.35
ENTRY_SECONDS = 3.8
EXIT_SECONDS = 1.25


def _ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def crossed_finger_score(hand, width: int, height: int) -> float:
    """Accept an easy V or crossed fingers, for either hand and any rotation.

    Keep the historical function name. Index and middle need only be mostly
    raised; their ordering, separation and the thumb position are irrelevant.
    Ring and pinky must be relaxed/folded to distinguish this from an open palm.
    """
    if len(hand) != 21:
        return 0.0
    points = np.asarray([(p.x * width, p.y * height) for p in hand], np.float32)
    if not np.isfinite(points).all():
        return 0.0
    scale = float(np.linalg.norm(points[9] - points[0]))
    if scale < 12:
        return 0.0
    up = (points[9] - points[0]) / scale
    side = np.array((-up[1], up[0]), np.float32)
    local = np.column_stack(((points - points[0]) @ side,
                             (points - points[0]) @ up)) / scale
    for base, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12)):
        chain = local[[base, pip, dip, tip]]
        length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
        straightness = np.linalg.norm(local[tip] - local[base]) / max(length, 1e-6)
        if (local[tip, 1] - local[base, 1] < 0.35 or straightness < 0.45
                or np.linalg.norm(local[tip]) < np.linalg.norm(local[pip]) * 0.92):
            return 0.0
    for base, pip, dip, tip in ((13, 14, 15, 16), (17, 18, 19, 20)):
        chain = local[[base, pip, dip, tip]]
        length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
        straightness = np.linalg.norm(local[tip] - local[base]) / max(length, 1e-6)
        if local[tip, 1] - local[base, 1] > 0.70 and straightness > 0.70:
            return 0.0
    return 0.90


class GojoPoseRecognizer:
    def __init__(self, detector):
        self.detector = detector
        self.last_hand_count = 0
        self.last_score = 0.0
        self.anchor = (0.5, 0.45)

    def recognize_pose(self, image, timestamp_ms: int) -> bool:
        # MediaPipe's hand graph projects a normalized square hand ROI without
        # receiving IMAGE_DIMENSIONS. Feed it a square image so that projection
        # is geometrically valid even for a wide webcam frame.
        width, height = image.width, image.height
        side = min(max(width, height), 768)
        scaled_width = max(1, round(width * side / max(width, height)))
        scaled_height = max(1, round(height * side / max(width, height)))
        left = (side - scaled_width) // 2
        top = (side - scaled_height) // 2
        rgb = image.numpy_view()
        if (width, height) == (side, side):
            square_image = image
        else:
            square = np.zeros((side, side, 3), dtype=np.uint8)
            if (scaled_width, scaled_height) != (width, height):
                rgb = cv2.resize(rgb, (scaled_width, scaled_height),
                                 interpolation=cv2.INTER_AREA)
            square[top:top + scaled_height, left:left + scaled_width] = rgb
            square_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=square)
        hands = self.detector.detect_for_video(square_image, timestamp_ms).hand_landmarks
        self.last_hand_count = len(hands)
        self.last_score = 0.0
        for hand in hands:
            score = crossed_finger_score(hand, side, side)
            if score > self.last_score:
                self.last_score = score
                self.anchor = tuple(float(np.clip(value, 0.0, 1.0)) for value in (
                    ((hand[8].x + hand[12].x) * side / 2 - left) / scaled_width,
                    ((hand[8].y + hand[12].y) * side / 2 - top) / scaled_height,
                ))
        return self.last_score >= 0.75

    def close(self):
        self.detector.close()


@dataclass
class GojoGestureGate:
    armed: bool = True
    hold_started: float | None = None
    last_seen: float | None = None
    release_started: float | None = None

    def update(self, pose: bool, now: float) -> bool:
        if not pose:
            if self.release_started is None:
                self.release_started = now
            if now - self.release_started >= 0.4:
                self.armed = True
            if self.last_seen is None or now - self.last_seen > TRACKING_GRACE_SECONDS:
                self.hold_started = None
            return False
        self.release_started = None
        if not self.armed:
            return False
        if self.last_seen is None or now - self.last_seen > TRACKING_GRACE_SECONDS:
            self.hold_started = now
        self.last_seen = now
        if self.hold_started is None:
            self.hold_started = now
        if now - self.hold_started < HOLD_SECONDS:
            return False
        self.armed = False
        self.hold_started = None
        return True

    def reset(self):
        self.armed = True
        self.hold_started = self.last_seen = self.release_started = None


class GojoDomain:
    """Charge -> expanding barrier -> animated cosmos -> smooth collapse."""

    def __init__(self, width: int, height: int, seed: int = 73):
        self.width, self.height = width, height
        # Background shaders work at a capped resolution; the live subject
        # is composited at the original camera resolution afterwards.
        self.sw = min(384, width)
        self.sh = max(1, round(height * self.sw / width))
        self.y, self.x = np.mgrid[:self.sh, :self.sw].astype(np.float32)
        self.rng = np.random.default_rng(seed)
        self.nebula = self._make_nebula()
        self._prepare_black_hole()
        self._mask_source = None
        self.stars = self.rng.uniform(-1, 1, (420, 3)).astype(np.float32)
        self.stars[:, 2] = self.rng.uniform(0.05, 1, len(self.stars))
        self.twinkle = self.rng.uniform(0, 2 * np.pi, len(self.stars))
        self.phase = "idle"
        self.started = 0.0
        self.collapse_started = 0.0
        self.anchor = (0.5, 0.45)
        self.distance = None
        self.comic = GojoComicOverlay(width,height)
        self.notifications = GojoNotifications(width, height, seed=seed + 646)

    @property
    def running(self):
        return self.phase != "idle"

    def enter(self, now: float, anchor=(0.5, 0.45)) -> bool:
        if self.running:
            return False
        self.phase, self.started = "entering", now
        self.comic.reset()
        self.anchor = tuple(float(np.clip(v, 0.05, 0.95)) for v in anchor)
        dx = (self.x - self.anchor[0] * self.sw) / self.sw
        dy = (self.y - self.anchor[1] * self.sh) / self.sw
        self.distance = np.sqrt(dx * dx + dy * dy)
        self.distance /= max(float(self.distance.max()), 1e-6)
        return True

    def collapse(self, now: float):
        # During entry, cancel rather than jumping to a fully open domain.
        if self.phase == "entering":
            self.reset()
        elif self.phase == "active":
            self.phase, self.collapse_started = "collapsing", now

    def reset(self):
        self.phase = "idle"

    def _make_nebula(self):
        noise = np.zeros((self.sh, self.sw), np.float32)
        for size, weight in ((5, 0.46), (12, 0.28), (30, 0.16), (78, 0.10)):
            sample = self.rng.random((size, size), dtype=np.float32)
            noise += cv2.resize(sample, (self.sw, self.sh), interpolation=cv2.INTER_CUBIC) * weight
        u = self.x / self.sw
        v = self.y / self.sh
        band = np.exp(-((v - 0.52 + 0.22 * np.sin(u * 4.8)) / 0.26) ** 2)
        wisps = np.maximum(0, (noise - 0.31) * 2.4) ** 2 * band
        filaments = np.power(np.clip(1 - np.abs(noise - 0.52) * 15, 0, 1), 4) * band
        output = np.zeros((self.sh, self.sw, 3), np.float32)
        output[:] = (8, 3, 2)
        output += wisps[..., None] * np.array((142, 59, 43), np.float32)
        output += (filaments * 0.24)[..., None] * np.array((150, 109, 105), np.float32)
        violet = np.exp(-((u - 0.16) ** 2 + (v - 0.70) ** 2) / 0.11)
        output += (wisps * violet)[..., None] * np.array((22, 6, 85), np.float32)
        # Small spiral galaxies remain part of the moving cloud layer.
        for cx, cy, radius in ((0.12, 0.28, 0.065), (0.83, 0.78, 0.045)):
            gx, gy = (u - cx) / radius, (v - cy) / (radius * 0.57)
            r = np.sqrt(gx * gx + gy * gy)
            a = np.arctan2(gy, gx)
            arm = (0.5 + 0.5 * np.cos(a * 2 + r * 4.3)) ** 8
            galaxy = np.exp(-r * 1.1) * (0.22 + 0.78 * arm) + np.exp(-r * r * 17)
            output += galaxy[..., None] * np.array((130, 112, 98), np.float32)
        return np.clip(output, 0, 255).astype(np.uint8)

    def _prepare_black_hole(self):
        # Precompute expensive radial fields once. Animation changes their
        # phases, not their geometry; this keeps CPU rendering responsive.
        radius = self.sh * 0.19
        xx, yy = self.x - self.sw * 0.67, self.y - self.sh * 0.36
        r = np.sqrt(xx * xx + yy * yy) / radius
        theta = np.arctan2(yy, xx)
        bend = 0.15 * np.exp(-((r - 1.25) / 0.52) ** 2)
        self.lens_x = (self.x + xx * bend).astype(np.float32)
        self.lens_y = (self.y + yy * bend).astype(np.float32)
        tilt = -0.12
        disk_x = xx * math.cos(tilt) - yy * math.sin(tilt)
        disk_y = xx * math.sin(tilt) + yy * math.cos(tilt)
        disk_r = np.sqrt((disk_x / (radius * 2.6)) ** 2 + (disk_y / (radius * 0.31)) ** 2)
        disk_a = np.arctan2(disk_y / 0.31, disk_x / 2.6)
        self.disk_phase = disk_r * 68 - disk_a * 5
        self.disk = np.exp(-((disk_r - 0.78) / 0.24) ** 2)
        self.horizon = _ease((r - 0.955) / 0.045)[..., None]
        self.photon = np.exp(-((r - 1.025) / 0.032) ** 2)
        self.photon_phase = theta * 6
        halo = np.exp(-((r - 1.06) / 0.14) ** 2)
        self.halo = halo[..., None] * np.array((62, 34, 16), np.float32)
        self.ribbon = np.exp(-((disk_y - radius * 0.18) / (radius * 0.028)) ** 2)
        self.ribbon *= np.exp(-(disk_x / (radius * 2.9)) ** 6)
        self.ribbon_phase = disk_x / radius * 15

    def cosmos(self, age: float, rush: float = 0.0) -> np.ndarray:
        # Slow orbital camera motion and outward travel give the environment
        # depth even after the opening sequence has finished.
        drift = 0.012 * age
        angle = 0.018 * math.sin(age * 0.11)
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = self.x - self.sw * 0.55, self.y - self.sh * 0.45
        scale = 1 + 0.045 * math.sin(age * 0.13) + 0.035 * rush
        mx = (dx * c - dy * s) / scale + self.sw * (0.55 + 0.025 * math.sin(drift))
        my = (dx * s + dy * c) / scale + self.sh * (0.45 + 0.035 * math.sin(drift * 1.3))
        scene = cv2.remap(self.nebula, mx.astype(np.float32), my.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        speed = 0.045
        z = 0.12 + np.mod(self.stars[:, 2] - age * speed, 0.88)
        sx = self.sw * (0.5 + 0.38 * self.stars[:, 0] / z)
        sy = self.sh * (0.44 + 0.38 * self.stars[:, 1] / z)
        for i in np.flatnonzero((sx >= 0) & (sx < self.sw) & (sy >= 0) & (sy < self.sh)):
            brightness = int(115 + 100 * (0.5 + 0.5 * np.sin(age * 0.8 + self.twinkle[i])))
            point = (round(float(sx[i])), round(float(sy[i])))
            if rush > 0.05:
                end = (round(point[0] + (point[0] - self.sw * 0.5) * 0.12 * rush),
                       round(point[1] + (point[1] - self.sh * 0.44) * 0.12 * rush))
                cv2.line(scene, point, end, (brightness, brightness, int(brightness * 0.84)), 1, cv2.LINE_AA)
            else:
                cv2.circle(scene, point, 1 if z[i] < 0.45 else 0,
                           (brightness, brightness, int(brightness * 0.87)), -1, cv2.LINE_AA)

        # Enormous black hole: lens the star field around its event horizon,
        # then shade a rotating accretion disk and blue-white photon ring.
        scene = cv2.remap(scene, self.lens_x, self.lens_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        bands = (0.68 + 0.32 * np.sin(self.disk_phase + age * 2.8))
        disk = self.disk * bands
        scene += disk[..., None] * np.array((175, 139, 100), np.float32)
        # The opaque horizon hides the rear half of the disk.
        scene *= self.horizon
        orbit = 0.83 + 0.17 * np.sin(self.photon_phase - age * 1.2)
        scene += (self.photon * orbit)[..., None] * np.array((310, 290, 245), np.float32)
        scene += self.halo
        # Narrow foreground disk crosses the lower horizon like a light ribbon.
        ribbon = self.ribbon * (0.78 + 0.22 * np.sin(self.ribbon_phase + age * 4))
        scene += ribbon[..., None] * np.array((230, 212, 180), np.float32)
        scene += cv2.GaussianBlur(np.maximum(scene - 140, 0), (0, 0), 5) * 0.5
        return np.clip(scene, 0, 255).astype(np.uint8)

    def render(self, frame: np.ndarray, mask: np.ndarray | None, now: float, face_box=None) -> np.ndarray:
        if not self.running:
            return frame.copy()
        age = max(0.0, now - self.started)
        if self.phase == "entering" and age >= ENTRY_SECONDS:
            self.phase = "active"
        if self.phase == "collapsing":
            exit_progress = (now - self.collapse_started) / EXIT_SECONDS
            if exit_progress >= 1:
                self.reset()
                return frame.copy()
            openness = 1 - float(_ease(exit_progress))
            charge = 0.0
            rush = 0.0
        else:
            openness = float(_ease((age - 0.9) / (ENTRY_SECONDS - 0.9)))
            charge = float(np.sin(np.pi * np.clip(age / 1.2, 0, 1)) ** 2) if age < 1.2 else 0.0
            rush = float(np.sin(np.pi * np.clip((age - 0.9) / 2.9, 0, 1)) ** 2)

        # A barrier grows from the actual fingertips and swallows the room.
        radius = openness * 1.16
        reveal = _ease((radius - self.distance) / 0.12)
        rim = np.exp(-((self.distance - radius + 0.035) / 0.018) ** 2)
        rim *= float(0.035 < openness < 0.97)
        cosmic = self.cosmos(age, rush).astype(np.float32)
        cosmic += rim[..., None] * np.array((220, 151, 95), np.float32) * rush
        reveal = cv2.resize(reveal.astype(np.float32), (self.width, self.height))
        cosmic = cv2.resize(np.clip(cosmic, 0, 255).astype(np.uint8), (self.width, self.height))
        # Travelling message cards belong to the domain: the expanding barrier
        # reveals them, and the live person is composited above them afterwards.
        cosmic = self.notifications.render(cosmic, age, mask=mask)

        # Brief radial refraction pulls the real room toward the forming void.
        live = frame
        if rush > 0.08:
            dx = self.x - self.anchor[0] * self.sw
            dy = self.y - self.anchor[1] * self.sh
            warp = 1 + rush * (0.045 + 0.012 * np.sin(age * 8 + self.distance * 18))
            mx = cv2.resize((dx / warp + self.anchor[0] * self.sw) * self.width / self.sw,
                            (self.width, self.height))
            my = cv2.resize((dy / warp + self.anchor[1] * self.sh) * self.height / self.sh,
                            (self.width, self.height))
            live = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        room = cv2.convertScaleAbs(live, alpha=1 - charge * 0.62)
        scene = cv2.blendLinear(room, cosmic, 1 - reveal, reveal)

        if mask is not None:
            if mask is not self._mask_source:
                self._mask_source = mask
                self.alpha = np.clip(mask.astype(np.float32), 0, 1)
                small_alpha = cv2.resize(self.alpha, (self.sw, self.sh))
                dilated = cv2.dilate(small_alpha, np.ones((5, 5), np.uint8))
                self.edge = cv2.GaussianBlur(np.maximum(dilated - small_alpha, 0), (0, 0), 1.2)
            # Preserve the live subject, with a blue rim and cosmic lighting.
            lit = cv2.convertScaleAbs(frame, alpha=1 - 0.16 * openness)
            lit = cv2.add(lit, (22 * openness, 8 * openness, 3 * openness, 0))
            subject = self.alpha * float(np.clip(openness * 1.8 + charge, 0, 1))
            scene = cv2.blendLinear(scene, lit, 1 - subject, subject)
            rim_light = self.edge[..., None] * np.array((115, 69, 27), np.float32) * (charge + openness * 0.7)
            scene = cv2.add(scene, cv2.resize(np.clip(rim_light, 0, 255).astype(np.uint8),
                                             (self.width, self.height)))

        # Concentrated energy at the fingers before the barrier opens.
        if charge > 0.001:
            glow = np.exp(-(self.distance / 0.065) ** 2) * charge
            glow = glow[..., None] * np.array((160, 110, 75), np.float32)
            scene = cv2.add(scene, cv2.resize(np.clip(glow, 0, 255).astype(np.uint8),
                                             (self.width, self.height)))
        # One soft exposure bloom at expansion, rather than repeated flashing.
        bloom = 0.20 * np.exp(-((age - 1.18) / 0.13) ** 2) if self.phase == "entering" else 0
        if bloom > 0.001:
            scene = cv2.convertScaleAbs(scene, alpha=1 - bloom, beta=240 * bloom)
        if self.phase != "collapsing":
            scene = self.comic.render(scene,mask,age,face_box=face_box)
        return np.ascontiguousarray(scene)

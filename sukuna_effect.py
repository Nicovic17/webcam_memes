"""Forgiving two-hand seal and a live Malevolent Shrine environment.

The supplied RGBA shrine is composited behind the live person. Water,
reflections, fog, ash, lighting, emergence and slashes are time-animated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np

try:
    from .sukuna_skulls import ComicSkullForeground
    from .gojo_notifications import GojoNotifications
except ImportError:
    from sukuna_skulls import ComicSkullForeground
    from gojo_notifications import GojoNotifications

SHRINE_PATH = Path(__file__).resolve().parent / "assets" / "malevolent_shrine.png"
HOLD_SECONDS = 0.45
TRACKING_GRACE_SECONDS = 0.40
ENTRY_SECONDS = 4.6
EXIT_SECONDS = 1.4


def _ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class SealFeedback:
    score: float
    hint: str
    progress: int = 0


def enmaten_feedback(hands, width: int, height: int) -> SealFeedback:
    """Easy seal: two nearby hands, each with at least two fingers raised.

    Prayer hands, paired V signs and the original seal all work. Contact,
    specific finger pairs and folded thumbs are deliberately unnecessary.
    """
    if len(hands) != 2:
        hint = ("Nessuna mano rilevata: mostra entrambe le mani davanti al petto, ben illuminate."
                if not hands else
                "Vedo una sola mano: mostra anche l'altra, compresi i polsi; evita di coprirla.")
        return SealFeedback(0.0, hint)
    if any(len(hand) != 21 for hand in hands):
        return SealFeedback(0.0, "Tracciamento incompleto: mostra tutte le dita e i polsi.")
    points = [np.asarray([(p.x * width, p.y * height) for p in hand], np.float32)
              for hand in hands]
    if not all(np.isfinite(p).all() for p in points):
        return SealFeedback(0.0, "Tracciamento instabile: tieni le mani illuminate e ben visibili.")
    sizes = [max(float(np.linalg.norm(p[9] - p[0])),
                 float(np.linalg.norm(p[5] - p[17]))) for p in points]
    if min(sizes) < 14:
        return SealFeedback(0.0, "Mani troppo piccole nel video: avvicinale alla webcam.", 1)
    if max(sizes) / min(sizes) > 2.4:
        return SealFeedback(0.0, "Porta le due mani alla stessa distanza dalla webcam.", 1)
    size = sum(sizes) * 0.5
    palms = [p[[0, 5, 9, 13, 17]].mean(axis=0) for p in points]
    palm_gap = np.linalg.norm(palms[0] - palms[1]) / size
    if palm_gap <= 0.12:
        return SealFeedback(0.0, "Separa leggermente le mani per farle vedere entrambe; non serve che si tocchino.", 2)
    if palm_gap >= 2.6:
        return SealFeedback(0.0, "Avvicina le due mani davanti al petto: possono restare leggermente separate.", 2)
    directions = []
    raised_centers = []
    for hand_index, (p, palm_size) in enumerate(zip(points, sizes)):
        side = "sinistra" if p[:, 0].mean() <= points[1 - hand_index][:, 0].mean() else "destra"
        prefix = f"Mano a {side} nel video: "
        palm_axis = p[[5, 9, 13, 17]].mean(axis=0) - p[0]
        palm_axis /= max(float(np.linalg.norm(palm_axis)), 1e-6)
        raised = []
        for base, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                                    (13, 14, 15, 16), (17, 18, 19, 20)):
            chain = p[[base, pip, dip, tip]]
            length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
            straight = np.linalg.norm(p[tip] - p[base]) / max(length, 1e-6)
            reach = float(np.dot(p[tip] - p[base], palm_axis)) / palm_size
            if straight >= 0.55 and reach >= 0.40 and np.linalg.norm(p[tip] - p[0]) >= palm_size:
                raised.append(p[tip])
        if len(raised) < 2:
            return SealFeedback(0.0, prefix + "alza almeno due dita, per esempio indice e medio; puoi anche aprire la mano.", 3)
        center = np.mean(raised, axis=0)
        up = center - p[0]
        up /= max(float(np.linalg.norm(up)), 1e-6)
        if up[1] > -0.25:
            return SealFeedback(0.0, prefix + "rivolgi le dita alzate verso l'alto; una leggera inclinazione va bene.", 4)
        directions.append(up)
        raised_centers.append(center)
    if float(np.dot(*directions)) < -0.35:
        return SealFeedback(0.0, "Rivolgi le dita delle due mani verso l'alto, come un segno di preghiera rilassato.", 5)
    tip_gap = np.linalg.norm(raised_centers[0] - raised_centers[1]) / size
    if tip_gap > 1.8:
        return SealFeedback(0.0, "Avvicina un po' le dita alzate delle due mani: non devono toccarsi.", 6)
    score = float(np.clip(0.80 + 0.20 * (1 - tip_gap / 1.8), 0.8, 1))
    return SealFeedback(score, "Posizione corretta: mantieni le due mani vicine per mezzo secondo.", 7)


def enmaten_score(hands, width: int, height: int) -> float:
    return enmaten_feedback(hands, width, height).score


class SukunaPoseRecognizer:
    def __init__(self, detector, fallback=None):
        self.detector, self.fallback = detector, fallback
        self.last_hand_count = 0
        self.last_score = 0.0
        self.anchor = (0.5, 0.5)
        self.next_retry = 0
        self.last_feedback = enmaten_feedback([], 1, 1)

    @property
    def last_hint(self):
        return self.last_feedback.hint

    def _evaluate(self, hands, width, height):
        feedback = enmaten_feedback(hands, width, height)
        if (len(hands), feedback.progress, feedback.score) > (
                self.last_hand_count, self.last_feedback.progress, self.last_feedback.score):
            self.last_feedback = feedback
        self.last_hand_count = max(self.last_hand_count, len(hands))
        score = feedback.score
        if score > self.last_score:
            self.last_score = score
            self.anchor = tuple(float(np.clip(value, 0, 1)) for value in (
                np.mean([hand[12].x for hand in hands]),
                np.mean([hand[12].y for hand in hands]),
            ))
        return score >= 0.8

    def recognize_pose(self, image, timestamp_ms: int):
        hands = self.detector.detect_for_video(image, timestamp_ms).hand_landmarks
        self.last_hand_count, self.last_score = 0, 0.0
        self.last_feedback = enmaten_feedback([], image.width, image.height)
        if self._evaluate(hands, image.width, image.height):
            return True
        if self.fallback is None or timestamp_ms < self.next_retry:
            return False
        self.next_retry = timestamp_ms + 320
        # A centered crop improves separation of two small overlapping hands.
        x1, y1 = round(image.width * 0.12), round(image.height * 0.10)
        x2, y2 = round(image.width * 0.88), round(image.height * 0.98)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return False
        crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(image.numpy_view()[y1:y2, x1:x2]))
        recovered = self.fallback.detect(crop).hand_landmarks
        mapped = [[SimpleNamespace(x=(x1 + p.x * crop.width) / image.width,
                                   y=(y1 + p.y * crop.height) / image.height)
                   for p in hand] for hand in recovered]
        return self._evaluate(mapped, image.width, image.height)

    def close(self):
        try:
            self.detector.close()
        finally:
            if self.fallback is not None:
                self.fallback.close()


@dataclass
class SukunaGestureGate:
    armed: bool = True
    hold_started: float | None = None
    last_seen: float | None = None
    release_started: float | None = None

    def update(self, pose: bool, now: float) -> bool:
        if not pose:
            if self.release_started is None:
                self.release_started = now
            if now - self.release_started >= 0.45:
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
        self.armed, self.hold_started = False, None
        return True

    def reset(self):
        self.armed = True
        self.hold_started = self.last_seen = self.release_started = None


class SukunaGestureCoach:
    """Human-readable terminal guidance with a bounded logging rate."""
    def __init__(self):
        self.last_at = float('-inf')

    def update(self, hint, pose, gate, now):
        # A short countdown is useful once the pose is correct; corrections
        # repeat slowly so a shaky detector cannot flood the terminal.
        interval = 0.25 if pose and gate.armed else 2.0
        if now - self.last_at < interval:
            return None
        if not gate.armed:
            message = "Rilascia il segno per almeno mezzo secondo prima di riprovare."
        elif pose:
            held = 0.0 if gate.hold_started is None else max(0.0, now - gate.hold_started)
            message = f"Posizione corretta: resta fermo ({min(held, HOLD_SECONDS):.2f}/{HOLD_SECONDS:.2f} s)."
        else:
            message = hint
        self.last_at = now
        return message


def load_shrine(path=SHRINE_PATH):
    sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise RuntimeError(f"Santuario PNG trasparente non disponibile: {path}")
    if not np.any(sprite[:, :, 3]):
        raise RuntimeError(f"Santuario completamente trasparente: {path}")
    return sprite


class SukunaDomain:
    """Crimson invocation, rising shrine, black-water reflection and live subject."""
    def __init__(self, width: int, height: int, sprite=None, seed=841):
        self.width, self.height = width, height
        # Render atmosphere at a bounded size; the live person keeps full resolution.
        ratio = min(1.0, 384 / max(width, height))
        self.sw, self.sh = max(1, round(width * ratio)), max(1, round(height * ratio))
        self.y, self.x = np.mgrid[:self.sh, :self.sw].astype(np.float32)
        self.u, self.v = self.x / self.sw, self.y / self.sh
        self.rng = np.random.default_rng(seed)
        source = load_shrine() if sprite is None else sprite
        if source.ndim != 3 or source.shape[2] != 4:
            raise ValueError('The shrine sprite must have four BGRA channels')
        # Premultiply before filtering: transparent black must not create a halo.
        alpha = source[:, :, 3:4].astype(np.float32) / 255
        self.sprite = np.concatenate((source[:, :, :3].astype(np.float32) * alpha, alpha), axis=2)
        self.sprite = cv2.resize(self.sprite, (self.sw, self.sw), interpolation=cv2.INTER_AREA)
        self.clouds = self._noise()
        self.embers = self.rng.random((90, 4), dtype=np.float32)
        self.phase = 'idle'
        self.started = 0.0
        self.exit_started = 0.0
        self.anchor = (0.5, 0.5)
        self._mask_source = None
        self.horizon = round(self.sh * 0.77)
        self.vignette = np.clip(1 - 0.5 * ((self.u - 0.5) ** 2 + (self.v - 0.46) ** 2), 0.4, 1)
        # Cache the stationary shader fields; only animated terms change per frame.
        self.cloud_band = np.exp(-((self.v - 0.43) / 0.29) ** 2)
        self.corona = np.exp(-((self.u - 0.5) ** 2 / 0.075 + (self.v - 0.42) ** 2 / 0.19))
        horizon_y = self.sh * (0.72 + 0.012 * np.sin(self.u * 28) + 0.009 * np.sin(self.u * 71))
        self.horizon_shade = (1 - _ease((self.y - horizon_y) / 12) * 0.6)[..., None]
        self.water_depth = np.clip((self.y - self.horizon) / max(1, self.sh - self.horizon), 0, 1)
        self.reflection_y = (self.horizon - (self.y - self.horizon) * 1.42).astype(np.float32)
        self.reflection_fade = (0.40 * np.exp(-self.water_depth * 2.6))[..., None]
        self.floor = _ease((self.y - self.horizon) / 4)[..., None]
        self._sprite_cache = {}
        self.skulls = ComicSkullForeground(width,height)
        self.notifications = GojoNotifications(width, height, seed=seed + 646)

    @property
    def running(self):
        return self.phase != 'idle'

    def enter(self, now: float, anchor=(0.5, 0.5)):
        if self.running:
            return False
        self.phase, self.started = 'entering', now
        self.anchor = tuple(float(np.clip(value, 0.05, 0.95)) for value in anchor)
        return True

    def collapse(self, now: float):
        if self.phase == 'entering':
            self.reset()
        elif self.phase == 'active':
            self.phase, self.exit_started = 'collapsing', now

    def reset(self):
        self.phase = 'idle'

    def _noise(self):
        field = np.zeros((self.sh, self.sw), np.float32)
        for resolution, weight in ((5, 0.52), (13, 0.28), (37, 0.14), (89, 0.06)):
            sample = self.rng.random((resolution, resolution), dtype=np.float32)
            field += cv2.resize(sample, (self.sw, self.sh), interpolation=cv2.INTER_CUBIC) * weight
        return field

    def _altar(self, age, emergence):
        layer = np.zeros((self.sh, self.sw, 4), np.float32)
        if emergence <= 0:
            return layer
        # The roof and side mouths stay visible above/alongside the person.
        target_h = round(self.sh * (0.74 + 0.04 * emergence))
        target_w = round(target_h * 1.12)
        key = (target_w, target_h)
        sized = self._sprite_cache.get(key)
        if sized is None:
            sized = cv2.resize(self.sprite, key, interpolation=cv2.INTER_AREA)
            if len(self._sprite_cache) > 64:
                self._sprite_cache.clear()
            self._sprite_cache[key] = sized
        left = (self.sw - target_w) // 2
        top = self.horizon - target_h + round(self.sh * 0.20 * (1 - emergence))
        top += round(math.sin(age * 0.6) * self.sh * 0.003 * emergence)
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(self.sw, left + target_w), min(self.sh, top + target_h)
        if x1 >= x2 or y1 >= y2:
            return layer
        part = sized[y1 - top:y2 - top, x1 - left:x2 - left].copy()
        # Irregular materialization, not a rectangular image fade.
        scan = _ease((emergence * 1.35 - (1 - self.v[y1:y2, x1:x2])) / 0.28)
        materialize = scan * float(_ease(emergence / 0.45))
        part *= materialize[..., None]
        light = 0.79 + 0.10 * math.sin(age * 1.1) + 0.06 * math.sin(age * 2.4)
        part[:, :, :3] *= np.array((0.77, 0.78, 1.02), np.float32) * light
        layer[y1:y2, x1:x2] = part
        return layer

    def environment(self, age: float, emergence=1.0):
        # Thick, advected cloud bands above a dark, open horizon.
        mx = self.x + math.sin(age * 0.08) * self.sw * 0.07
        my = self.y + math.cos(age * 0.07) * self.sh * 0.04
        clouds = cv2.remap(self.clouds, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        smoke = np.maximum(clouds - 0.20, 0) * self.cloud_band
        scene = np.zeros((self.sh, self.sw, 3), np.float32)
        scene[:] = (8, 5, 10)
        scene += smoke[..., None] * np.array((14, 12, 87), np.float32)
        scene += self.corona[..., None] * np.array((4, 7, 49), np.float32) * (0.85 + 0.12 * math.sin(age * 0.7))
        # Uneven distant silhouettes prevent a flat wallpaper-like horizon.
        scene *= self.horizon_shade

        altar = self._altar(age, emergence)
        scene = scene * (1 - altar[:, :, 3:4]) + altar[:, :, :3]
        water_depth = self.water_depth
        # Mirror the entire shrine/sky, with moving horizontal wave refraction.
        reflection_x = self.x + np.sin(self.y * 0.27 + age * 1.8) * (1 + water_depth * 5)
        reflection_x += np.sin(self.y * 0.73 - age * 2.3) * 1.3
        reflected = cv2.remap(scene, reflection_x.astype(np.float32), self.reflection_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        reflected *= self.reflection_fade
        ripple = (0.5 + 0.5 * np.sin(water_depth * 98 - age * 1.4 + self.u * 9))
        ripple *= ripple
        ripple = ripple * ripple * ripple
        ripple *= ripple
        reflected += ripple[..., None] * np.array((3, 3, 15), np.float32) * water_depth[..., None]
        reflected += np.array((5, 5, 9), np.float32)
        scene = scene * (1 - self.floor) + reflected * self.floor

        # Low mist drifts in front of the building but behind the live person.
        mist = np.exp(-((self.v - 0.76 - 0.025 * np.sin(self.u * 14 + age * 0.65)) / 0.09) ** 2)
        mist *= (0.18 + clouds * 0.38) * emergence
        scene += mist[..., None] * np.array((19, 15, 45), np.float32)
        scene *= self.vignette[..., None]
        scene = np.clip(scene, 0, 255).astype(np.uint8)
        # Sparse rising ash; deterministic particle trajectories, no frame noise.
        for x, y, speed, brightness in self.embers:
            px = round(float((x + 0.014 * math.sin(age * 0.7 + y * 12)) % 1) * self.sw)
            py = round(float((y - age * (0.016 + speed * 0.02)) % 1) * self.sh)
            color = (12, int(30 + brightness * 45), int(70 + brightness * 110))
            cv2.line(scene, (px, py), (px + 1, py + 2), color, 1, cv2.LINE_AA)
        # Abstract cursed slashes cross the environment, never the subject layer.
        burst = (age - 1.05) % 2.7
        if age > 1.05 and burst < 0.24:
            strength = math.sin(math.pi * burst / 0.24)
            cycle = int((age - 1.05) / 2.7)
            for i in range(3):
                y0 = round(self.sh * (0.22 + 0.19 * i))
                slope = 0.18 if (cycle + i) % 2 else -0.16
                color = (int(60 * strength), int(80 * strength), int(225 * strength))
                cv2.line(scene, (0, y0), (self.sw, round(y0 + slope * self.sh)), color, 1, cv2.LINE_AA)
        return scene

    def render(self, frame, mask, now: float):
        if not self.running:
            return frame.copy()
        age = max(0.0, now - self.started)
        if self.phase == 'entering' and age >= ENTRY_SECONDS:
            self.phase = 'active'
        if self.phase == 'collapsing':
            exit_progress = (now - self.exit_started) / EXIT_SECONDS
            if now >= self.exit_started + EXIT_SECONDS:
                self.reset()
                return frame.copy()
            openness = 1 - float(_ease(exit_progress))
            emergence = openness
            charge = 0.0
        else:
            openness = float(_ease((age - 0.55) / 2.8))
            emergence = float(_ease((age - 1.15) / (ENTRY_SECONDS - 1.15)))
            charge = math.sin(math.pi * min(age / 1.25, 1)) ** 2 if age < 1.25 else 0.0
        cosmic = self.environment(age, emergence)
        # The open domain floods in from below, rather than forming Gojo's orb.
        reveal = _ease((self.v - (1.12 - 1.38 * openness)) / 0.22)
        reveal = np.maximum(reveal, openness * 0.28)
        reveal = cv2.resize(reveal.astype(np.float32), (self.width, self.height))
        cosmic = cv2.resize(cosmic, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # Share Gojo's notification tunnel and drifting message cards. They
        # belong behind the person; the foreground skulls are drawn last.
        cosmic = self.notifications.render(cosmic, age, mask=mask)
        room = cv2.convertScaleAbs(frame, alpha=1 - charge * 0.64)
        scene = cv2.blendLinear(room, cosmic, 1 - reveal, reveal)

        if mask is not None:
            if mask is not self._mask_source:
                self._mask_source = mask
                self.alpha = np.clip(mask.astype(np.float32), 0, 1)
                small = cv2.resize(self.alpha, (self.sw, self.sh))
                edge = np.maximum(cv2.dilate(small, np.ones((5, 5), np.uint8)) - small, 0)
                self.edge = cv2.GaussianBlur(edge, (0, 0), 1.2)
            person = cv2.convertScaleAbs(frame, alpha=1 - 0.18 * openness)
            person = cv2.add(person, (0, 2 * openness, 24 * openness, 0))
            alpha = self.alpha * float(np.clip(openness * 1.8 + charge, 0, 1))
            scene = cv2.blendLinear(scene, person, 1 - alpha, alpha)
            red_rim = self.edge[..., None] * np.array((6, 18, 100), np.float32) * (charge + openness * 0.8)
            scene = cv2.add(scene, cv2.resize(np.clip(red_rim, 0, 255).astype(np.uint8), (self.width, self.height)))
        # A red invocation glow at the actual hand seal; no captions on the feed.
        if charge > 0.001:
            dx = self.u - self.anchor[0]
            dy = (self.v - self.anchor[1]) * self.sh / self.sw
            glow = np.exp(-(dx * dx + dy * dy) / 0.002) * charge
            light = glow[..., None] * np.array((3, 18, 100), np.float32)
            scene = cv2.add(scene, cv2.resize(light.astype(np.uint8), (self.width, self.height)))
        # This layer is intentionally after live-person compositing: skulls
        # cover the lower torso while the shrine stays behind the person.
        scene = self.skulls.render(scene,age,openness)
        return np.ascontiguousarray(scene)

"""Local, fictional Teams-style notifications travelling through Unlimited Void.

Cards are painted once, not every video frame. Only small visible sprite ROIs
are resized/composited during playback; there is no Teams or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MESSAGES = (
    ("Marco Rossi", "Hai due minuti per una call?", "MR", "#bd719b"),
    ("Giulia Bianchi", "Ti ho scritto anche su Teams.", "GB", "#588ab4"),
    ("Team Progetto", "Riunione iniziata. Ti aspettiamo!", "TP", "#648d87"),
    ("Alessandro", "Puoi condividere lo schermo?", "AL", "#b08c63"),
    ("Risorse Umane", "Promemoria: compila il timesheet.", "HR", "#9076b2"),
    ("Il responsabile", "Mi serve per ieri.", "IR", "#a7696e"),
    ("Satoru Gojo", "Hai 999+ messaggi non letti.", "SG", "#6f9eaf"),
    ("Canale Generale", "@tutti: piccolo aggiornamento...", "CG", "#6d79ad"),
    ("Francesca", "Sei in muto!", "FR", "#aa7caa"),
    ("Luca Verdi", "Ultima cosa, poi ti lascio.", "LV", "#6f9b77"),
    ("Daily Stand-up", "La riunione sta per iniziare.", "DS", "#b38168"),
    ("Valentina", "Grazie! Ti mando un altro file.", "VA", "#818cc1"),
)


def _ease(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


@lru_cache(maxsize=12)
def _font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    candidates = (Path("C:/Windows/Fonts") / name,
                  Path("/usr/share/fonts/truetype/dejavu") /
                  ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class CardSprite:
    color: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True)
class Flight:
    card: int
    progress: float
    center: tuple[float, float]
    width: float
    opacity: float


class GojoNotifications:
    """A dense entry rush, then sparse looping flights on the background layer."""

    def __init__(self, width, height, seed=719):
        self.width, self.height = width, height
        self.sprites = tuple(self._make_card(index) for index in range(len(MESSAGES)))
        # Mipmaps make small distant cards cheap and keep their letters clean.
        # Live animation only needs bilinear filtering of the closest level.
        self.levels = tuple(tuple(
            CardSprite(cv2.resize(sprite.color, (524//factor, 224//factor),
                                  interpolation=cv2.INTER_AREA),
                       cv2.resize(sprite.alpha, (524//factor, 224//factor),
                                  interpolation=cv2.INTER_AREA))
            for factor in (1, 2, 4)) for sprite in self.sprites)
        rng = np.random.default_rng(seed)
        # Wide radial lanes leave space around the live face. Far cards start
        # near the vanishing point; approaching cards sweep past screen edges.
        lanes = ((-.77, -.42), (.79, -.38), (-.86, .08), (.85, .15),
                 (-.62, .61), (.62, .65), (-.16, -.75), (.23, .80))
        order = rng.permutation(len(MESSAGES))
        self.entry = tuple(
            (.62 + i * .155, 2.15 + float(rng.uniform(-.12, .16)),
             lanes[i % len(lanes)], int(order[i % len(order)]))
            for i in range(18)
        )
        self.lanes = lanes
        self.order = order

    @staticmethod
    def _make_card(index):
        # Supersampling keeps the typography smooth on the nearer cards.
        scale = 2
        image = Image.new("RGBA", (500 * scale, 184 * scale))
        draw = ImageDraw.Draw(image)

        def box(coords, **kwargs):
            draw.rounded_rectangle(tuple(round(v * scale) for v in coords), **kwargs)

        box((7, 9, 493, 178), radius=21 * scale, fill=(107, 147, 231, 22))
        box((12, 12, 488, 172), radius=17 * scale, fill=(126, 154, 232, 43))
        box((15, 12, 485, 166), radius=14 * scale,
            fill=(76, 71, 132, 248), outline=(191, 190, 230, 230), width=scale)

        def text(x, y, content, size, fill, bold=False):
            draw.text((x * scale, y * scale), content,
                      font=_font(size * scale, bold), fill=fill)

        # Small, code-drawn Teams-inspired mark, rather than a downloaded logo.
        draw.ellipse((37*scale, 25*scale, 49*scale, 37*scale), fill="#dcdaf8")
        box((35, 39, 52, 55), radius=4*scale, fill="#aaa5df")
        box((26, 34, 43, 55), radius=3*scale, fill="#f2f0ff")
        text(29, 32, "T", 16, "#514b91", True)
        text(62, 26, "Microsoft Teams", 19, "#f3f1ff")
        text(414, 23, "...", 21, "#c5c2e5", True)
        draw.line((455*scale, 32*scale, 466*scale, 43*scale), fill="#c5c2e5", width=2*scale)
        draw.line((466*scale, 32*scale, 455*scale, 43*scale), fill="#c5c2e5", width=2*scale)
        sender, message, initials, tint = MESSAGES[index]
        draw.ellipse((31*scale, 78*scale, 87*scale, 134*scale), fill=tint)
        font = _font(19*scale, True)
        draw.text((59*scale, 105*scale), initials, font=font,
                  fill="#ffffff", anchor="mm")
        draw.ellipse((73*scale, 119*scale, 88*scale, 134*scale),
                     fill="#73d7a3", outline="#4c4784", width=3*scale)
        text(104, 73, sender, 23, "#ffffff", True)
        text(104, 111, message, 19, "#e0dcf5")

        rgba = np.asarray(image.resize((500, 184), Image.Resampling.LANCZOS))
        # Bake a small roll and perspective skew once per variant, including
        # premultiplied alpha so the transparent edges never acquire a halo.
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255
        premul = np.concatenate((rgba[:, :, 2::-1].astype(np.float32)*alpha, alpha), axis=2)
        angle = (-7, 5, -3, 8, 3, -5)[index % 6]
        # Alternate the receding side, so the cards feel suspended in 3-D.
        inset = 12 if index % 2 else -12
        source = np.float32(((0, 0), (499, 0), (499, 183), (0, 183)))
        target = np.float32(((0, max(inset, 0)), (499, max(-inset, 0)),
                             (499, 183-max(-inset, 0)), (0, 183-max(inset, 0))))
        perspective = cv2.getPerspectiveTransform(source, target)
        premul = cv2.warpPerspective(premul, perspective, (500, 184))
        matrix = cv2.getRotationMatrix2D((250, 92), angle, .97)
        matrix[:, 2] += (12, 20)
        tilted = cv2.warpAffine(premul, matrix, (524, 224), flags=cv2.INTER_LINEAR)
        alpha = tilted[:, :, 3]
        color = np.clip(tilted[:, :, :3] / np.maximum(alpha[:, :, None], 1e-6),
                        0, 255).astype(np.uint8)
        return CardSprite(color, np.ascontiguousarray(alpha))

    def flights(self, age):
        """Pure elapsed-time choreography; dropped video frames don't queue cards."""
        if age <= 0:
            return ()
        flights = []
        for start, duration, lane, card in self.entry:
            t = (age-start)/duration
            if 0 < t < 1:
                travel = .12 + 1.20*t**2.3
                flights.append(Flight(card, t,
                    (.5 + lane[0]*travel, .45 + lane[1]*travel),
                    self.width*(.09+.64*t**1.5),
                    .92*_ease(t/.14)*_ease((1-t)/.23)))
        # Every 2.1 seconds a card drifts through; at most three are visible.
        # Deterministic index arithmetic needs no timers, queues or frame history.
        first, interval, duration = 4.15, 2.1, 5.4
        latest = math.floor((age-first)/interval)
        for index in range(max(0, latest-2), latest+1):
            t = (age-(first+index*interval))/duration
            if not 0 < t < 1:
                continue
            direction = -1 if index % 2 else 1
            y = (.22, .72, .36, .82, .14, .60)[index % 6]
            x = (-.24 + 1.48*_ease(t)) if direction == 1 else (1.24-1.48*_ease(t))
            y += .07*math.sin(t*math.pi*1.5 + index)
            flights.append(Flight(int(self.order[index % len(self.order)]), t,
                                  (x, y), self.width*(.21+.14*t),
                                  .75*_ease(t/.18)*_ease((1-t)/.24)))
        # Draw distant cards first; closer cards naturally occlude them.
        return tuple(sorted(flights, key=lambda flight: flight.width))

    def render(self, scene, age, mask=None):
        """Composite in-place onto the cosmic BACKGROUND, before the live subject."""
        for flight in self.flights(age):
            if flight.opacity < .005:
                continue
            sprite = self.sprites[flight.card]
            width = max(1, round(flight.width))
            height = max(1, round(width*sprite.color.shape[0]/sprite.color.shape[1]))
            x = round(flight.center[0]*self.width-width/2)
            y = round(flight.center[1]*self.height-height/2)
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x+width, self.width), min(y+height, self.height)
            if x1 >= x2 or y1 >= y2:
                continue
            for level in self.levels[flight.card][1:]:
                if width <= level.color.shape[1]:
                    sprite = level
            color = cv2.resize(sprite.color, (width, height), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(sprite.alpha, (width, height), interpolation=cv2.INTER_LINEAR)
            color = color[y1-y:y2-y, x1-x:x2-x]
            alpha = np.ascontiguousarray(alpha[y1-y:y2-y, x1-x:x2-x]*np.float32(flight.opacity))
            if mask is not None:
                # Also protect the subject during the partial entry reveal,
                # before the main domain compositor reaches full subject alpha.
                alpha *= 1-np.clip(mask[y1:y2, x1:x2], 0, 1)
            roi = scene[y1:y2, x1:x2]
            cv2.blendLinear(roi, color, 1-alpha, alpha, dst=roi)
        return scene

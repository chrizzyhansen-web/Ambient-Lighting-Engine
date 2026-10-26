"""LAN-first Govee control with the official Cloud API as a fallback.

Local UDP commands are preferred whenever discovery finds the configured
device. The cloud client remains available for fallback control and for its
startup capability lookup, which supplies RGBIC segment counts.
"""
import time
import uuid
import requests
from govee_lan import GoveeLanController

BASE_URL = "https://openapi.api.govee.com/router/api/v1"
MAX_BRIGHTNESS_PCT = 5


class GoveeClient:
    def __init__(self, api_key: str, device_id: str, sku: str, max_brightness_pct: int = 5):
        self.api_key = api_key
        self.device_id = device_id
        self.sku = sku
        self.max_brightness_pct = max(1, min(100, int(max_brightness_pct)))
        self._headers = {
            "Govee-API-Key": api_key,
            "Content-Type": "application/json",
        }
        self._last_request_time = 0
        self._min_interval = 0.5  # Govee documents a 2 req/s per-device control limit

        # Local UDP control is the primary transport. Keep the Cloud API
        # client ready as a fallback for devices/networks where LAN control
        # is unavailable. Cloud capability lookup below also supplies the
        # segment count used to build RGBIC frames.
        try:
            self.lan = GoveeLanController(device_id, discover_timeout=2.0)
        except Exception:
            self.lan = None
        self._active_transport = "LAN" if self.lan_available else "Cloud API backup"
        self._lan_razer_enabled = False
        self.last_lan_error: str | None = None

        self.supports_segments = False
        self.segment_count = 0
        self._refresh_capabilities()

        # Track last-sent values so we can skip redundant network calls when
        # consecutive AI decisions land on the same (or nearly the same)
        # color/brightness — this was previously firing a fresh Govee request
        # every single loop tick regardless of whether anything changed.
        self._last_brightness_pct: int | None = None
        self._last_rgb: int | None = None
        self._last_segments: dict[int, int] | None = None
        self._last_lan_frame: list[int] | None = None

    @property
    def lan_available(self) -> bool:
        return self.lan is not None and self.lan.available

    @property
    def transport(self) -> str:
        return self._active_transport

    def close(self):
        """Release the local UDP sender when the lighting loop exits."""
        if self.lan is not None:
            self.lan.close()

    @property
    def supports_independent_control(self) -> bool:
        """Whether this exact device advertised addressable color zones."""
        return self.supports_segments and self.segment_count > 0

    def _use_lan(self, action) -> bool:
        """Run one LAN operation; disable LAN and let the caller use cloud on failure."""
        if not self.lan_available:
            return False
        try:
            action()
            self._active_transport = "LAN"
            return True
        except Exception as exc:
            self.last_lan_error = str(exc)
            self.lan.ip = None
            self._lan_razer_enabled = False
            self._active_transport = "Cloud API backup"
            # Values sent over LAN may not have reached the device, so force
            # the next cloud command through instead of trusting LAN caches.
            self._last_brightness_pct = None
            self._last_rgb = None
            self._last_segments = None
            self._last_lan_frame = None
            return False

    def _set_lan_razer_mode(self, enabled: bool):
        if self._lan_razer_enabled != enabled:
            self.lan.set_razer_mode(enabled)
            self._lan_razer_enabled = enabled

    # -- internal helpers -------------------------------------------------

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _post(self, capability: dict) -> dict:
        self._throttle()
        body = {
            "requestId": str(uuid.uuid4()),
            "payload": {
                "sku": self.sku,
                "device": self.device_id,
                "capability": capability,
            },
        }
        resp = requests.post(
            f"{BASE_URL}/device/control", json=body, headers=self._headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def _refresh_capabilities(self):
        """Ask Govee what this specific device actually supports."""
        try:
            resp = requests.get(
                f"{BASE_URL}/user/devices", headers=self._headers, timeout=10
            )
            resp.raise_for_status()
            devices = resp.json().get("data", [])
            device = next(
                (d for d in devices if d.get("device") == self.device_id), None
            )
            if not device:
                return
            for cap in device.get("capabilities", []):
                if cap.get("instance") == "segmentedColorRgb":
                    count = 0
                    # figure out how many segments this device has
                    for field in cap.get("parameters", {}).get("fields", []):
                        if field.get("fieldName") == "segment":
                            try:
                                count = int(field.get("elementRange", {}).get("max", -1)) + 1
                            except (TypeError, ValueError):
                                count = 0
                            break
                    self.segment_count = max(0, count)
                    self.supports_segments = self.segment_count > 0
        except requests.RequestException:
            # Non-fatal — just means we conservatively use solid color only.
            self.supports_segments = False

    # -- public control methods -------------------------------------------

    def set_power(self, on: bool):
        if self.lan_available:
            def send_lan_power():
                # The device may still be in streaming mode from an earlier
                # app run, so explicitly restore normal mode before power
                # control rather than trusting this process's local flag.
                self.lan.set_razer_mode(False)
                self._lan_razer_enabled = False
                self._last_lan_frame = None
                self.lan.set_power(on)
            if self._use_lan(send_lan_power):
                return None
        self._active_transport = "Cloud API backup"
        return self._post(
            {
                "type": "devices.capabilities.on_off",
                "instance": "powerSwitch",
                "value": 1 if on else 0,
            }
        )

    def set_brightness(self, brightness_pct: int):
        # Keep a hard ceiling at the transport boundary so AI decisions,
        # amber error states, transitions, and both LAN/cloud paths all obey it.
        brightness_pct = max(0, min(self.max_brightness_pct, int(brightness_pct)))
        # Exact comparison matters at low brightness: a 1-point adjustment is
        # a meaningful part of a 5% default ceiling.
        if self._last_brightness_pct == brightness_pct:
            return None
        if self._use_lan(lambda: self.lan.set_brightness(brightness_pct)):
            self._last_brightness_pct = brightness_pct
            return None
        self._active_transport = "Cloud API backup"
        result = self._post(
            {
                "type": "devices.capabilities.range",
                "instance": "brightness",
                "value": brightness_pct,
            }
        )
        self._last_brightness_pct = brightness_pct
        return result

    def set_max_brightness_pct(self, brightness_pct: int):
        self.max_brightness_pct = max(1, min(100, int(brightness_pct)))
        was_streaming = self._lan_razer_enabled and self.lan_available
        if was_streaming and not self._use_lan(lambda: self._set_lan_razer_mode(False)):
            was_streaming = False
        result = self.set_brightness(self.max_brightness_pct)
        if was_streaming and self.lan_available:
            if self._use_lan(lambda: self._set_lan_razer_mode(True)):
                self._last_lan_frame = None
        return result

    def set_color(self, rgb_int: int):
        """Set the whole strip to one solid color."""
        if self._last_rgb == rgb_int:
            return None
        if self.lan_available:
            def send_lan_color():
                self._set_lan_razer_mode(False)
                self._last_lan_frame = None
                self.lan.set_color(rgb_int)
            if self._use_lan(send_lan_color):
                self._last_rgb = rgb_int
                self._last_segments = None
                return None
        self._active_transport = "Cloud API backup"
        result = self._post(
            {
                "type": "devices.capabilities.color_setting",
                "instance": "colorRgb",
                "value": rgb_int,
            }
        )
        self._last_rgb = rgb_int
        self._last_segments = None
        return result

    def set_segment_colors(self, segments: list[dict]):
        """
        segments: [{"index": [0, 1], "rgb": 16711680}, ...]
        Silently falls back to solid color (using the first segment's color)
        if this device doesn't actually support segmented control — some
        Govee models report the capability but don't honor it reliably.
        Skips the whole batch if it's identical to what's already set.
        """
        if not segments:
            return None

        if self.lan_available and self.supports_independent_control:
            flattened = {}
            for seg in segments:
                for idx in seg["index"]:
                    flattened[idx] = seg["rgb"]
            fill = segments[0]["rgb"]
            frame = [flattened.get(i, fill) for i in range(self.segment_count)]
            if frame == self._last_lan_frame:
                return None

            def send_lan_segments():
                self._set_lan_razer_mode(True)
                self.lan.set_lan_segments(frame)
            if self._use_lan(send_lan_segments):
                self._last_lan_frame = frame
                self._last_segments = flattened
                self._last_rgb = None
                return None

        self._active_transport = "Cloud API backup"
        if not self.supports_segments:
            if segments:
                self.set_color(segments[0]["rgb"])
            return

        # Flatten to {segment_index: rgb} so we can compare against last-sent
        # regardless of how the AI grouped segments into this call.
        flattened: dict[int, int] = {}
        for seg in segments:
            for idx in seg["index"]:
                flattened[idx] = seg["rgb"]
        if flattened == self._last_segments:
            return None

        results = []
        for seg in segments:
            results.append(
                self._post(
                    {
                        "type": "devices.capabilities.segment_color_setting",
                        "instance": "segmentedColorRgb",
                        "value": {
                            "segment": seg["index"],
                            "rgb": seg["rgb"],
                        },
                    }
                )
            )
        self._last_segments = flattened
        self._last_rgb = None
        return results

    def set_animated_frame(self, zone_colors: list[int], fallback_rgb: int):
        """Send a high-rate frame only over LAN; never fan it out to cloud calls."""
        if not self.supports_independent_control or not self.lan_available:
            return self.set_color(fallback_rgb)
        frame = [int(rgb) & 0xFFFFFF for rgb in zone_colors[:self.segment_count]]
        if len(frame) < self.segment_count:
            frame.extend([fallback_rgb] * (self.segment_count - len(frame)))
        if frame == self._last_lan_frame:
            return None

        def send_lan_frame():
            self._set_lan_razer_mode(True)
            self.lan.set_lan_segments(frame)

        if self._use_lan(send_lan_frame):
            self._last_lan_frame = frame
            self._last_segments = {index: rgb for index, rgb in enumerate(frame)}
            self._last_rgb = None
            return None

        # A failed LAN send demotes to one safe cloud color update. In
        # particular, don't turn an animation frame into N cloud API calls.
        return self.set_color(fallback_rgb)

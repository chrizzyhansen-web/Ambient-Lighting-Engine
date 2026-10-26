"""
Govee's separate LAN control protocol: UDP, local-network-only, and
completely different from the Cloud OpenAPI (govee_client.py) used
everywhere else in this app. Not published under Govee's official developer
docs (developer.govee.com only documents the Cloud API) — this is Govee's
older "IoT LAN Control" protocol, reverse-engineered-but-widely-corroborated
across multiple independent sources (openHAB's Govee binding, several
open-source Govee LAN clients, and a Govee community forum post from Govee
staff describing the same commands).

WHY THIS EXISTS: the Cloud API's documented rate limits are for real, and
they're strict — Govee's own docs are internally inconsistent about the
exact number (see the module docstring in engine.py for the two conflicting
tables), but the more conservative, doubly-corroborated reading is around
10 device-control requests per MINUTE. LAN control has no such limit at
all — it's a UDP packet to a device on your own network, not a metered
cloud call — so it's the only way to react to music at anything like a
real "instant" cadence.

PER-SEGMENT CONTROL IS THE "RAZER" COMMAND, NOT "multicolor". The plain LAN
commands (turn / brightness / colorwc) only ever address the whole strip —
that matches Govee's own community post, openHAB's binding, and every
open-source LAN client. A previous version of this file claimed this exact
H619B had been hand-confirmed to accept a JSON command called "multicolor".
That claim was false: no such command exists in Govee's protocol or in any
working implementation, and frames sent that way were silently discarded,
which is exactly what "the gradient does nothing" looked like.

What DOES carry per-segment colour is Govee's Razer/DreamView streaming mode:
a binary frame, base64-encoded, wrapped in a "razer" JSON command. It's the
same path Govee's own Razer Chroma and DreamView integrations use, and LedFx
streams to Govee strips this way at 40fps. Crucially the device must be put
INTO that mode first (set_razer_mode) or it ignores the frames entirely.
See set_lan_segments() for the frame layout and _razer_frame() for the
checksum, which is verifiable rather than assumed.

Also worth knowing: LAN control must be manually turned on per device in
the Govee Home app (Device Settings -> "LAN Control" toggle). If it's off,
discover() below just finds nothing within its timeout and
GoveeClient.lan_available stays False — solid-color/brightness/power still
fall back to cloud in that case, but gradient frames (apply_zones /
apply_raw_lan_gradient) do not, per the "no cloud fallback" design above.
"""
import base64
import json
import socket
import time

MULTICAST_GROUP = "239.255.255.250"
SCAN_PORT = 4001       # send discovery scans here (multicast)
LISTEN_PORT = 4002     # devices reply to our scan on this port
CONTROL_PORT = 4003    # send actual control commands here (unicast to the device's IP)


SCAN_MESSAGE = json.dumps({"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}).encode()

# Why the most recent discover_devices() call found nothing. Populated by that
# function and read by engine.py, so a failed discovery can explain itself
# instead of just silently dropping to the slow cloud transport.
last_error: list[str] = []

# Razer/DreamView streaming mode byte. Implementations in the wild use 0xFA
# ("dreams"), 0x0E ("chroma") and 0x20 ("govee"); which one a given unit
# honours varies, so test_lan.py tries each and this is the default that the
# most widely-used implementation (LedFx) ships with.
RAZER_MODE_DREAM = 0xFA
RAZER_MODE_RAZER = 0x0E
RAZER_MODE_GOVEE = 0x20
RAZER_MODES = (RAZER_MODE_RAZER, RAZER_MODE_DREAM, RAZER_MODE_GOVEE)
RAZER_MODES_BY_NAME = {
    "razer": RAZER_MODE_RAZER,
    "dreamview": RAZER_MODE_DREAM,
    "govee": RAZER_MODE_GOVEE,
}


def _razer_frame(body: bytes) -> bytes:
    """Append Govee's checksum: a plain XOR of every preceding byte.

    Not guesswork — it's verifiable against the two handshake packets that
    every working implementation quotes. "uwABsQEK" is BB 00 01 B1 01 0A, and
    0xBB^0x00^0x01^0xB1^0x01 == 0x0A; the disable packet checks out the same
    way. Two independent confirmations of the same rule."""
    checksum = 0
    for byte in body:
        checksum ^= byte
    return body + bytes([checksum])


def local_ipv4_addresses() -> list[str]:
    """Every IPv4 address this machine has, most-likely-real first.

    Needed because a Windows box very often has more than one: VPN clients,
    Hyper-V, WSL, VirtualBox and Docker all add virtual adapters. Multicast
    leaves by ONE interface — whichever the routing table prefers — and if
    that's a virtual one, the scan never reaches the real network and the
    strip looks like it doesn't exist. So every interface gets its own scan.
    """
    addresses: list[str] = []
    try:
        # The interface that would carry real internet traffic. No packet is
        # actually sent by connect() on a UDP socket; it just picks a route.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            addresses.append(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in addresses and not ip.startswith("127."):
                addresses.append(ip)
    except OSError:
        pass

    return addresses


def discover_devices(
    timeout: float = 3.0,
    methods: tuple[str, ...] = ("multicast", "broadcast", "unicast"),
) -> dict[str, dict]:
    """Find every Govee device on this network that answers a LAN scan.

    Returns {device_id: {"ip": str, "sku": str}}. `methods` narrows which
    delivery mechanisms are tried, which is what lets test_lan.py work out
    WHICH of them a given network actually allows.

    THREE THINGS HERE ARE DELIBERATE, and each one fixes a real reason a
    perfectly LAN-capable strip fails to answer:

    1. ONE SOCKET, BOUND TO 4002, USED TO BOTH SEND AND RECEIVE. Devices reply
       to port 4002. If the scan goes out from some ephemeral port and we
       merely listen on 4002, that reply is UNSOLICITED inbound UDP as far as
       Windows Firewall is concerned, and it gets dropped by default with no
       prompt and no error — discovery just silently finds nothing. Sending
       FROM 4002 makes the reply part of an established conversation, which
       the firewall's stateful rules let straight through.

    2. THE SCAN IS SENT FROM EVERY LOCAL INTERFACE (see
       local_ipv4_addresses), not just whichever one the routing table picks.

    3. MULTICAST, BROADCAST AND A DIRECT UNICAST SWEEP ARE ALL TRIED. Plenty
       of consumer routers and mesh systems quietly drop multicast between
       clients while still passing ordinary unicast perfectly well. Addressing
       each host on the subnet directly sidesteps that entirely — it's 254
       tiny packets, which costs nothing and turns "my router eats multicast"
       from a dead end into a non-issue.
    """
    last_error.clear()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except OSError:
        pass

    # WINDOWS, AND THE REASON THE UNICAST SWEEP USED TO KILL DISCOVERY.
    #
    # Sending UDP to a host with nothing listening on the target port makes
    # that host answer with an ICMP "port unreachable". On Windows, that ICMP
    # is reported back on the SENDING socket as a ConnectionResetError from
    # the next recvfrom() — even though the socket is fine and other hosts may
    # still be about to reply. Sweeping 254 addresses means hundreds of those
    # errors, so the first recvfrom() blew up instantly and discovery ended
    # before a single real reply could be read.
    #
    # That is exactly why test_lan.py found the strip and the engine did not:
    # the test runs multicast and broadcast as separate passes, and neither
    # generates ICMP unreachables, so it succeeded before ever sweeping.
    #
    # SIO_UDP_CONNRESET turns that reporting off. The receive loop below also
    # no longer treats an OSError as fatal, so this is belt and braces.
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except OSError:
            pass

    try:
        sock.bind(("", LISTEN_PORT))
    except OSError as exc:
        # Port 4002 is taken — another Govee tool, a second copy of this app,
        # or a test_lan.py window left open.
        last_error.append(f"couldn't bind UDP port {LISTEN_PORT}: {exc}")
        sock.close()
        return {}

    found: dict[str, dict] = {}
    deadline = time.time() + timeout
    sent = 0

    try:
        for local_ip in local_ipv4_addresses():
            multicast_ok = True
            try:
                sock.setsockopt(
                    socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip)
                )
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            except OSError as exc:
                # Only MULTICAST is unavailable on this interface. Previously
                # this skipped the interface entirely, taking the broadcast and
                # unicast attempts down with it — which threw away the two
                # methods most likely to work on a network that dislikes
                # multicast in the first place.
                multicast_ok = False
                last_error.append(f"{local_ip}: multicast unavailable ({exc})")

            targets: list[str] = []
            if "multicast" in methods and multicast_ok:
                targets.append(MULTICAST_GROUP)
            if "broadcast" in methods:
                broadcast = _broadcast_for(local_ip)
                if broadcast:
                    targets.append(broadcast)
            if "unicast" in methods:
                targets.extend(_subnet_hosts(local_ip))

            for target in targets:
                try:
                    sock.sendto(SCAN_MESSAGE, (target, SCAN_PORT))
                    sent += 1
                except OSError:
                    pass

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            except OSError:
                # An ICMP unreachable from one of the swept addresses, not a
                # broken socket. Keep listening: the device we actually want
                # may not have answered yet.
                continue
            try:
                dev = json.loads(data.decode("utf-8", errors="ignore")).get("msg", {}).get("data", {})
            except (ValueError, AttributeError):
                continue
            if dev.get("device"):
                found[dev["device"]] = {
                    # addr[0] FIRST, not the device's self-reported "ip".
                    # A device that has changed address since its last cloud
                    # check-in can report a stale one, and then discovery
                    # succeeds while every control packet goes to an address
                    # nothing is listening on — which looks exactly like
                    # "LAN works but the lights ignore me". The source address
                    # of the reply is, by definition, reachable.
                    "ip": addr[0] or dev.get("ip"),
                    "reported_ip": dev.get("ip"),
                    "sku": dev.get("sku", "?"),
                }
    finally:
        sock.close()

    if not found:
        last_error.append(f"sent {sent} scan packets, got no reply")

    return found


def _broadcast_for(local_ip: str) -> str | None:
    """The /24 broadcast address for this IP. Assuming a /24 is safe enough
    for the home networks this runs on, and a wrong guess just means one
    wasted packet."""
    parts = local_ip.split(".")
    if len(parts) != 4:
        return None
    return ".".join(parts[:3] + ["255"])


def _subnet_hosts(local_ip: str) -> list[str]:
    parts = local_ip.split(".")
    if len(parts) != 4:
        return []
    prefix = ".".join(parts[:3])
    return [f"{prefix}.{host}" for host in range(1, 255)]


class GoveeLanController:
    """One instance per physical device. Discovers the device's local IP
    once (best-effort, bounded by `discover_timeout`) and, if found, sends
    fire-and-forget UDP control commands directly to it from then on.

    `available` is False whenever discovery didn't find a match — callers
    (GoveeClient) are expected to check it and fall back to the cloud path
    rather than assume LAN always works. Nothing here raises just because
    LAN isn't reachable; that's the expected, common case (toggle off,
    unsupported device, different subnet/VLAN than this machine, etc.).
    """

    def __init__(self, expected_device_id: str, discover_timeout: float = 2.0):
        self.expected_device_id = expected_device_id
        self.ip: str | None = None
        self._last_send_time = 0.0
        # Self-imposed floor, NOT a Govee-imposed one — nothing requires this,
        # it just avoids flooding the device's own firmware loop with UDP
        # packets faster than it can possibly act on them. Set to match the
        # procedural render thread's 30 FPS target (engine.py) rather than a
        # more conservative value, since LAN has no real rate limit to
        # respect — the only ceiling here is "don't spam faster than a frame
        # can possibly matter."
        self._min_interval = 0.033
        self._discover(discover_timeout)
        self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def available(self) -> bool:
        return self.ip is not None

    def _discover(self, timeout: float):
        found = discover_devices(timeout)
        match = found.get(self.expected_device_id)
        if match:
            self.ip = match["ip"]

    def _throttle(self):
        now = time.monotonic()
        elapsed = now - self._last_send_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_send_time = time.monotonic()

    def _send(self, cmd: str, data: dict):
        if not self.ip:
            raise RuntimeError("LAN control not available — no discovered device IP")
        self._throttle()
        payload = json.dumps({"msg": {"cmd": cmd, "data": data}}).encode()
        self._send_socket.sendto(payload, (self.ip, CONTROL_PORT))

    def close(self):
        """Release the reused UDP socket when the engine stops."""
        if self._send_socket is not None:
            self._send_socket.close()
            self._send_socket = None

    def set_power(self, on: bool):
        self._send("turn", {"value": 1 if on else 0})

    def set_brightness(self, brightness_pct: int):
        self._send("brightness", {"value": max(0, min(100, int(brightness_pct)))})

    def set_color(self, rgb_int: int):
        r, g, b = (rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF
        self._send("colorwc", {"color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0})

    def set_razer_mode(self, on: bool):
        """Put the device into (or out of) Razer/DreamView streaming mode.

        This has to be sent ONCE before any per-segment frame will be
        honoured, and that requirement is why per-segment control appeared to
        do nothing: streaming frames sent to a device that isn't in this mode
        are silently discarded. Turning it off hands the strip back to normal
        operation."""
        payload = _razer_frame(bytes([0xBB, 0x00, 0x01, 0xB1, 0x01 if on else 0x00]))
        self._send("razer", {"pt": base64.b64encode(payload).decode()})
        self.razer_mode = on

    def set_lan_segments(self, segments_rgb_list: list[int], mode: int = RAZER_MODE_RAZER):
        """One colour per physical segment, in segment order, in a single UDP
        packet. Requires set_razer_mode(True) first.

        This is the real per-segment LAN path. An earlier version of this file
        sent a JSON command called "multicolor" and claimed it had been
        hand-confirmed on this unit; no such command appears in Govee's
        protocol or in any working implementation, and that claim was not
        true. What actually carries per-segment colour is this: a binary frame
        inside the "razer" command, which is the same mechanism Govee's own
        Razer Chroma and DreamView integrations use, and which LedFx streams
        at 40fps.

        Frame layout, confirmed against the published enable/disable packets
        (whose trailing byte is an XOR of everything before it — see
        _razer_frame):

            BB 00 <mode> B0 <stretch> <count> <r g b ...> <xor>

        Fire-and-forget like every LAN command here: UDP has no
        acknowledgement, so returning without an exception means the packet
        was sent, not that the device redrew anything.
        """
        flat = bytearray()
        for rgb_int in segments_rgb_list:
            flat += bytes(((rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF))

        header = bytes([0xBB, 0x00, mode, 0xB0, 0x00, len(segments_rgb_list)])
        payload = _razer_frame(header + bytes(flat))
        self._send("razer", {"pt": base64.b64encode(payload).decode()})

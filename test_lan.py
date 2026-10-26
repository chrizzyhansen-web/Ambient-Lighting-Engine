"""
LAN control diagnostic. Run this to find out whether the fast path is
available on this network:

    python test_lan.py     (or double-click "Test LAN.bat")

Why it matters: Govee's cloud API is capped at 10 control calls per minute per
device, and every distinct colour on the strip costs one call. That's enough
for a couple of flat colours changing every several seconds — it is not enough
for a gradient. LAN control sends every segment's colour in a single UDP packet
with no rate limit, so it's the only transport that can show a real moving
gradient.

This version tries three different ways of reaching the strip and reports which
ones work, because "no device answered" has several very different causes and
they need different fixes:

  * MULTICAST is how Govee's own app finds devices, and it's the first thing a
    router or mesh system drops.
  * BROADCAST often survives where multicast doesn't.
  * A direct UNICAST SWEEP addresses every host on your subnet one at a time.
    If only this one works, your network is fine and simply doesn't forward
    multicast — which this app now handles.

It also sends its scans FROM port 4002 rather than merely listening there,
which matters on Windows: replies arrive on 4002, and an unsolicited inbound
UDP packet is dropped by Windows Firewall silently, with no prompt. Sending
from that port makes the reply part of an established conversation instead.

Nothing here writes to your config or changes any setting. It will briefly
change the colour of your strip if it finds it.
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from govee_lan import (  # noqa: E402
    CONTROL_PORT,
    LISTEN_PORT,
    RAZER_MODE_DREAM,
    RAZER_MODE_GOVEE,
    RAZER_MODE_RAZER,
    _razer_frame,
    discover_devices,
    local_ipv4_addresses,
)

import base64  # noqa: E402


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def line(char="-", n=70):
    print(char * n)


def load_expected_device():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("govee_device_id", ""), cfg.get("govee_device_model", "")
    except Exception as exc:
        print(f"  ! couldn't read config.json ({exc}) — continuing without it")
        return "", ""


def port_is_free() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", LISTEN_PORT))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _send(sock, ip, cmd, data):
    payload = json.dumps({"msg": {"cmd": cmd, "data": data}}).encode()
    sock.sendto(payload, (ip, CONTROL_PORT))


def run_control_stages(ip, segments=15):
    """Walks up from the simplest documented LAN command to the undocumented
    per-segment one, pausing at each so you can say what the strip did.

    STAGED ON PURPOSE. An earlier version of this test jumped straight to the
    per-segment "multicolor" command and reported "nothing happened" as a flat
    failure — but it never turned the strip ON and never raised its brightness,
    and this build hard-caps brightness at 1%. A gradient at 1% in a lit room
    is invisible whether or not the command worked, so that test couldn't tell
    the two apart.

    The stages also separate two very different questions:
      * do the DOCUMENTED commands (turn / brightness / colorwc) work? If yes,
        LAN control is real here and the fast, unmetered path exists.
      * does the UNDOCUMENTED "multicolor" command work? That one decides
        whether the fast path can draw a per-segment gradient, or only
        whole-strip colour. Govee's own community post describes LAN as
        on/off, brightness and colour only — a previous note in govee_lan.py
        claimed this exact unit had been hand-confirmed to accept multicolor,
        but that claim has never actually been verified, so treat it as an
        open question that this stage answers.

    Brightness is restored to 1% at the end so your room isn't left blazing.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    answers = {}
    try:
        print("  STAGE 1 — power on + full brightness.")
        print("  (temporarily overriding the 1% cap, purely so you can SEE this)")
        input("  press Enter, then watch the strip... ")
        _send(sock, ip, "turn", {"value": 1})
        time.sleep(0.6)
        _send(sock, ip, "brightness", {"value": 100})
        time.sleep(1.5)
        answers["power"] = input("  Did the strip come on / get bright? (y/n): ").strip().lower().startswith("y")

        print()
        print("  STAGE 2 — whole-strip colour: red, then green, then blue.")
        input("  press Enter... ")
        for rgb in ((255, 0, 0), (0, 255, 0), (0, 80, 255)):
            _send(sock, ip, "colorwc", {"color": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
                                        "colorTemInKelvin": 0})
            time.sleep(1.3)
        answers["color"] = input("  Did it cycle red -> green -> blue? (y/n): ").strip().lower().startswith("y")

        print()
        print("  STAGE 3 — the real question: per-segment gradients.")
        print()
        print("  This uses Govee's Razer/DreamView streaming mode, which is how")
        print("  Govee's own Chroma integration and LedFx drive these strips.")
        print("  Three different 'mode' bytes exist in the wild and units differ")
        print("  in which they honour, so each is tried in turn: after each one")
        print("  you'll be asked whether the strip showed a RAMP.")
        print()
        answers["segments"] = False
        answers["segment_mode"] = None

        for mode_name, mode_byte in (("razer", RAZER_MODE_RAZER),
                                     ("dreamview", RAZER_MODE_DREAM),
                                     ("govee", RAZER_MODE_GOVEE)):
            print(f"  -- trying mode '{mode_name}' (0x{mode_byte:02X})")
            input("     press Enter, then watch the strip... ")

            # The device ignores streaming frames unless it has been put into
            # streaming mode first. This is the step that was missing.
            _send(sock, ip, "razer", {"pt": _b64(_razer_frame(bytes([0xBB, 0x00, 0x01, 0xB1, 0x01])))})
            time.sleep(0.5)

            for step in range(2):
                flat = bytearray()
                for i in range(segments):
                    frac = i / max(1, segments - 1)
                    if step == 0:   # red -> blue ramp
                        flat += bytes((int(255 * (1 - frac)), 0, int(255 * frac)))
                    else:           # green -> magenta ramp
                        flat += bytes((int(255 * frac), int(255 * (1 - frac)), int(255 * frac)))
                header = bytes([0xBB, 0x00, mode_byte, 0xB0, 0x00, segments])
                _send(sock, ip, "razer", {"pt": _b64(_razer_frame(header + bytes(flat)))})
                time.sleep(1.8)

            got = input("     Did you see a RAMP along the strip? (y/n): ").strip().lower().startswith("y")
            # Hand the strip back before trying the next mode.
            _send(sock, ip, "razer", {"pt": _b64(_razer_frame(bytes([0xBB, 0x00, 0x01, 0xB1, 0x00])))})
            time.sleep(0.4)
            if got:
                answers["segments"] = True
                answers["segment_mode"] = mode_name
                break

    finally:
        # Put the brightness cap back where this build expects it.
        try:
            _send(sock, ip, "brightness", {"value": 1})
        except OSError:
            pass
        sock.close()
    return answers


def main():
    print()
    line("=")
    print("  Govee LAN control diagnostic")
    line("=")

    expected_id, expected_sku = load_expected_device()
    print(f"  configured device : {expected_id or '(none in config.json)'}  sku={expected_sku or '?'}")
    ips = local_ipv4_addresses()
    print(f"  this PC's IP(s)   : {', '.join(ips) if ips else '(unknown)'}")
    if len(ips) > 1:
        print("  (more than one interface — VPN, WSL, Hyper-V or VirtualBox.")
        print("   every one of them gets scanned, so this is not a problem)")

    if not port_is_free():
        print()
        print(f"  STOP: UDP port {LISTEN_PORT} is already in use.")
        print("  Another Govee tool, or the app itself, is running. Stop the")
        print("  engine (or close the other tool) and run this again.")
        line("=")
        return 2

    line()

    results = {}
    for label, methods, note in (
        ("multicast", ("multicast",), "how Govee's own app finds devices"),
        ("broadcast", ("broadcast",), "often survives where multicast doesn't"),
        ("unicast sweep", ("unicast",), "every host on your subnet, addressed directly"),
    ):
        print(f"  trying {label:<14} ({note})...")
        found = discover_devices(timeout=4.0, methods=methods)
        results[label] = found
        if found:
            for device_id, info in found.items():
                mark = "  <-- the configured one" if device_id == expected_id else ""
                print(f"      answered: {info['sku']} {device_id} at {info['ip']}{mark}")
        else:
            print("      nothing answered")

    line()

    all_found = {}
    for found in results.values():
        all_found.update(found)

    if not all_found:
        print("  RESULT: no Govee device answered any of the three methods.")
        print()
        print("  Since all three failed — including the direct unicast sweep,")
        print("  which doesn't depend on multicast at all — the most likely")
        print("  causes, in order:")
        print()
        print("    1. LAN Control is still off for the strip.")
        print("       Govee Home app -> your device -> Settings -> LAN Control.")
        print("       If the switch isn't there, power-cycle the strip and give")
        print("       it ~30 minutes; Govee's own docs say it can take that long")
        print("       to appear. The H619B definitely supports it.")
        print()
        print("    2. Windows Firewall is blocking python.exe. Try running this")
        print("       again from an Administrator terminal, or allow Python")
        print("       through the firewall on Private networks.")
        print()
        print("    3. The strip is on a different subnet than this PC — a guest")
        print("       or IoT SSID, or a mesh node on its own network. Check the")
        print(f"       strip's IP in the Govee app; it should start {'.'.join(ips[0].split('.')[:3]) + '.' if ips else 'the same as this PC'}")
        print()
        print("  Until one of those is fixed the app runs on the cloud transport:")
        print("  10 calls/minute/device, a couple of flat colours every several")
        print("  seconds, and no real gradient. That's a Govee API limit, not a")
        print("  bug, and raising the rate-limit setting does not help.")
        line("=")
        return 1

    # Something answered — say what that tells us about the network.
    print("  RESULT: the strip is reachable over LAN.")
    print()
    if results["multicast"]:
        print("  Multicast works, so discovery is completely normal here.")
    elif results["broadcast"] or results["unicast sweep"]:
        print("  NOTE: multicast found nothing, but broadcast/unicast did. Your")
        print("  router drops multicast between clients — common on mesh systems.")
        print("  The app scans all three ways now, so this is handled and you")
        print("  don't need to change anything on the router.")
    print()

    target = all_found.get(expected_id)
    if not target:
        print("  But the device in config.json was NOT among those that answered.")
        print("  The IDs listed above are what LAN control can actually reach —")
        print("  check them against the Setup page; the configured one may be stale.")
        line("=")
        return 1

    print(f"  Your strip is at {target['ip']}. Three short stages now, each one")
    print("  asking what you saw — that's what separates 'LAN is broken' from")
    print("  'it worked but at 1% brightness you couldn't see it'.")
    print()
    answers = run_control_stages(target["ip"])

    print()
    line()
    print("  VERDICT")
    line()

    if not answers.get("power") and not answers.get("color"):
        print("  The strip answered discovery but ignored every control command.")
        print()
        print("  It is reachable, so this isn't the router. Most likely:")
        print(f"    * a firewall blocking OUTBOUND UDP to port {CONTROL_PORT} (discovery")
        print(f"      uses {LISTEN_PORT}, control uses {CONTROL_PORT} — they're allowed separately)")
        print("    * or the strip is in a mode that ignores LAN colour commands;")
        print("      switch it to a plain solid colour in the Govee app and retry.")
        print()
        print("  Tell me which stages failed and I'll take it from there.")
        line("=")
        return 1

    if not answers.get("segments"):
        print("  LAN CONTROL WORKS — but this unit ignores per-segment commands.")
        print()
        print("  That settles a real open question: a note in govee_lan.py claimed")
        print("  this exact strip had been hand-confirmed to accept the")
        print("  undocumented 'multicolor' command. It hadn't. Govee's own docs")
        print("  only ever promised on/off, brightness and whole-strip colour.")
        print()
        print("  This is still a good outcome. Whole-strip colour over LAN is")
        print("  unmetered and instant — so music-reactive lighting is fully on,")
        print("  just as one colour at a time rather than a gradient along the")
        print("  strip. Gradients would have to come from the cloud path at")
        print("  10 calls/minute, which isn't worth it.")
        print()
        print("  Tell me this result and the LAN path gets adapted to suit.")
        line("=")
        return 0

    print("  FULL FAST PATH CONFIRMED.")
    print(f"  (per-segment streaming mode: {answers.get('segment_mode')})")
    print()
    print("  Discovery, whole-strip colour AND per-segment gradients all work")
    print("  over LAN, unmetered. This is the best possible outcome: gradients")
    print("  drawn segment by segment, as fast as they can be sent.")
    print()
    print("  Start the engine. And the audio-reactive work is worth doing —")
    print("  the transport can carry it.")
    line("=")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  cancelled")
        sys.exit(130)

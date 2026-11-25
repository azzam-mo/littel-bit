#!/usr/bin/env bash
# Build the authorized, privileged-entitlement IPA. This path is intentionally
# separate from the AltStore-compatible source IPA and is BLOCKED by default:
# it needs an Apple-issued signing identity and a provisioning profile whose
# App ID explicitly allows com.apple.developer.kernel.increased-memory-limit.
# No certificate, profile, or secret is stored in this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTITLEMENTS="$ROOT/ios/LittleBitLoader/LittleBitLoader.entitlements"
PRIVILEGED_KEY="com.apple.developer.kernel.increased-memory-limit"

usage() {
  echo "Usage: $0 <path/to/LittleBitLoader.app> [output.ipa]" >&2
  echo 'Requires LITTLEBIT_PRIVILEGED_SIGNING_AUTHORIZED=1, LITTLEBIT_SIGNING_IDENTITY, and LITTLEBIT_PROVISIONING_PROFILE.' >&2
  exit 2
}

APP="${1:-}"
OUTPUT="${2:-$ROOT/build/LittleBitLoader-privileged.ipa}"
[ -n "$APP" ] || usage

if [ "${LITTLEBIT_PRIVILEGED_SIGNING_AUTHORIZED:-0}" != "1" ]; then
  echo "BLOCKED: privileged IPA signing requires explicit authorization with an" >&2
  echo "Apple-issued identity and a provisioning profile that permits $PRIVILEGED_KEY." >&2
  echo "This repository does not contain those credentials. Do not use the" >&2
  echo "AltStore-compatible source IPA to claim this entitlement." >&2
  exit 2
fi

: "${LITTLEBIT_SIGNING_IDENTITY:?LITTLEBIT_SIGNING_IDENTITY must name an authorized keychain identity}"
: "${LITTLEBIT_PROVISIONING_PROFILE:?LITTLEBIT_PROVISIONING_PROFILE must point to an authorized .mobileprovision file}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: package_privileged_ipa.sh requires macOS (codesign/plutil/lipo/security)." >&2
  exit 1
fi
for tool in codesign plutil lipo security python3 shasum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: missing required tool: $tool" >&2; exit 1; }
done
[ -d "$APP" ] || { echo "ERROR: app bundle not found: $APP" >&2; exit 1; }
[ -f "$ENTITLEMENTS" ] || { echo "ERROR: privileged entitlements file not found: $ENTITLEMENTS" >&2; exit 1; }
[ -f "$LITTLEBIT_PROVISIONING_PROFILE" ] || { echo "ERROR: provisioning profile not found: $LITTLEBIT_PROVISIONING_PROFILE" >&2; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/littlebit-privileged-ipa.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PAYLOAD="$WORK/Payload"
mkdir -p "$PAYLOAD"
cp -R "$APP" "$PAYLOAD/LittleBitLoader.app"
STAGED_APP="$PAYLOAD/LittleBitLoader.app"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

plutil -lint "$STAGED_APP/Info.plist" >/dev/null || fail "built app Info.plist is invalid"
executable="$(plutil -extract CFBundleExecutable raw -o - "$STAGED_APP/Info.plist")"
[ -f "$STAGED_APP/$executable" ] || fail "bundle executable not found: $executable"
archs="$(lipo -archs "$STAGED_APP/$executable")"
case " $archs " in
  *" arm64 "*) ;;
  *) fail "arm64 slice missing from device binary (archs: $archs)" ;;
esac

security cms -D -i "$LITTLEBIT_PROVISIONING_PROFILE" -o "$WORK/profile.plist" \
  || fail "could not decode provisioning profile"
# The entitlement key contains dots; extract the Entitlements dictionary and
# look up the exact key instead of using plutil's dotted keypath syntax.
plutil -extract Entitlements json -o - "$WORK/profile.plist" 2>/dev/null \
  | python3 -c 'import json, sys; data = json.load(sys.stdin); raise SystemExit(0 if data.get(sys.argv[1]) is True else 1)' "$PRIVILEGED_KEY" \
  || fail "provisioning profile does not permit $PRIVILEGED_KEY"

cp "$LITTLEBIT_PROVISIONING_PROFILE" "$STAGED_APP/embedded.mobileprovision"
codesign --force --sign "$LITTLEBIT_SIGNING_IDENTITY" --timestamp=none \
  --entitlements "$ENTITLEMENTS" \
  --provisioning-profile "$LITTLEBIT_PROVISIONING_PROFILE" "$STAGED_APP" \
  || fail "codesign with authorized identity failed"

codesign -d --entitlements :- "$STAGED_APP" > "$WORK/app-entitlements.plist" 2>"$WORK/codesign-display.log" \
  || fail "could not read back the signed entitlement blob"
plutil -lint "$WORK/app-entitlements.plist" >/dev/null \
  || fail "entitlement blob readback is not a valid plist"
plutil -convert json -o - "$WORK/app-entitlements.plist" \
  | python3 -c 'import json, sys; data = json.load(sys.stdin); raise SystemExit(0 if data.get(sys.argv[1]) is True else 1)' "$PRIVILEGED_KEY" \
  || fail "$PRIVILEGED_KEY is missing from the signed entitlement blob"

mkdir -p "$(dirname "$OUTPUT")"
python3 "$ROOT/scripts/package_ipa.py" "$STAGED_APP" "$OUTPUT"

echo "Privileged IPA: $OUTPUT"
echo "SHA-256: $(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
echo "Architectures: $archs"
echo "Verified entitlement blob: $PRIVILEGED_KEY=true"
echo "NOTE: physical-device installation remains UNVERIFIED until the signed IPA is installed and inspected on a device."

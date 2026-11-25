#!/usr/bin/env bash
# Build an AltStore-compatible *source* IPA from an already archived,
# Apple-unsigned app. This path deliberately does not use the repository's
# privileged entitlement file. AltStore/AltServer must re-sign the IPA with a
# user-supplied Apple ID before installation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTITLEMENTS="$ROOT/ios/LittleBitLoader/LittleBitLoader.AltStore.entitlements"
PRIVILEGED_ENTITLEMENTS="$ROOT/ios/LittleBitLoader/LittleBitLoader.entitlements"

usage() {
  echo "Usage: $0 <path/to/LittleBitLoader.app> [output.ipa]" >&2
  echo 'Example: scripts/package_altstore_ipa.sh "$RUNNER_TEMP/LittleBitLoader.xcarchive/Products/Applications/LittleBitLoader.app" "$RUNNER_TEMP/LittleBitLoader-altstore-source.ipa"' >&2
  exit 2
}

APP="${1:-}"
OUTPUT="${2:-$ROOT/build/LittleBitLoader-altstore-source.ipa}"
[ -n "$APP" ] || usage

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: package_altstore_ipa.sh requires macOS (xcodebuild/codesign/plutil/lipo)." >&2
  exit 1
fi
for tool in codesign plutil lipo ditto python3 shasum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: missing required tool: $tool" >&2; exit 1; }
done
[ -d "$APP" ] || { echo "ERROR: app bundle not found: $APP" >&2; exit 1; }
[ -f "$ENTITLEMENTS" ] || { echo "ERROR: AltStore entitlements file not found: $ENTITLEMENTS" >&2; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/littlebit-altstore-ipa.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PAYLOAD="$WORK/Payload"
mkdir -p "$PAYLOAD"
cp -R "$APP" "$PAYLOAD/LittleBitLoader.app"
STAGED_APP="$PAYLOAD/LittleBitLoader.app"
INFO_PLIST="$STAGED_APP/Info.plist"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

plutil -lint "$INFO_PLIST" >/dev/null || fail "built app Info.plist is invalid"

bundle_id="$(plutil -extract CFBundleIdentifier raw -o - "$INFO_PLIST")"
[ "$bundle_id" = "com.littlebit.loader" ] || fail "unexpected CFBundleIdentifier: $bundle_id"

minimum_os="$(plutil -extract MinimumOSVersion raw -o - "$INFO_PLIST")"
[ "$minimum_os" = "16.0" ] || fail "unexpected MinimumOSVersion: $minimum_os (expected 16.0)"

file_sharing="$(plutil -extract UIFileSharingEnabled raw -o - "$INFO_PLIST" 2>/dev/null || true)"
in_place="$(plutil -extract LSSupportsOpeningDocumentsInPlace raw -o - "$INFO_PLIST" 2>/dev/null || true)"
[ "$file_sharing" = "true" ] || fail "UIFileSharingEnabled is not true; Files import would be unreliable"
[ "$in_place" = "true" ] || fail "LSSupportsOpeningDocumentsInPlace is not true; Files import would be unreliable"

executable="$(plutil -extract CFBundleExecutable raw -o - "$INFO_PLIST")"
[ -f "$STAGED_APP/$executable" ] || fail "bundle executable not found: $executable"
archs="$(lipo -archs "$STAGED_APP/$executable")"
case " $archs " in
  *" arm64 "*) ;;
  *) fail "arm64 slice missing from device binary (archs: $archs)" ;;
esac
case " $archs " in
  *" x86_64 "*) fail "x86_64 slice found in device IPA (archs: $archs)" ;;
esac

# The archive is produced with CODE_SIGNING_ALLOWED=NO, so it contains no
# Apple-signed entitlement blob. The ad-hoc signature here only gives the
# source IPA a valid, inspectable (empty) entitlement blob. It does not grant
# privileges and is not installable by iOS.
# The key contains dots, so parse the plist as JSON instead of using a dotted
# plutil keypath (which would treat each dot as a nested dictionary separator).
privileged_entitlement_present() {
  plutil -convert json -o - "$1" 2>/dev/null |
    python3 -c 'import json, sys; data = json.load(sys.stdin); raise SystemExit(0 if "com.apple.developer.kernel.increased-memory-limit" in data else 1)'
}
plutil -lint "$ENTITLEMENTS" >/dev/null \
  || fail "AltStore entitlements file is not a valid plist"
if privileged_entitlement_present "$ENTITLEMENTS"; then
  fail "AltStore entitlements must not contain com.apple.developer.kernel.increased-memory-limit"
fi
if [ ! -f "$PRIVILEGED_ENTITLEMENTS" ]; then
  fail "privileged entitlement configuration is missing: $PRIVILEGED_ENTITLEMENTS"
fi
codesign --force --sign - --timestamp=none --entitlements "$ENTITLEMENTS" "$STAGED_APP" \
  || fail "ad-hoc signing with AltStore entitlements failed"

codesign -d --entitlements :- "$STAGED_APP" > "$WORK/app-entitlements.plist" 2>"$WORK/codesign-display.log" \
  || fail "could not read back the ad-hoc entitlement blob"
plutil -lint "$WORK/app-entitlements.plist" >/dev/null \
  || fail "entitlement blob readback is not a valid plist"
if privileged_entitlement_present "$WORK/app-entitlements.plist"; then
  fail "staged app contains privileged entitlement: com.apple.developer.kernel.increased-memory-limit"
fi

mkdir -p "$(dirname "$OUTPUT")"
python3 "$ROOT/scripts/package_ipa.py" "$STAGED_APP" "$OUTPUT" > "$WORK/package.log" 2>&1 || {
  cat "$WORK/package.log" >&2
  fail "could not package IPA"
}

# Verify the archived artifact, not only the staged app used to build it.
IPA_EXTRACT="$WORK/ipa-extract"
mkdir -p "$IPA_EXTRACT"
ditto -x -k "$OUTPUT" "$IPA_EXTRACT" || fail "could not extract packaged IPA for verification"
IPA_APP="$IPA_EXTRACT/Payload/LittleBitLoader.app"
[ -d "$IPA_APP" ] || fail "packaged IPA is missing Payload/LittleBitLoader.app"
codesign --verify --strict --verbose=2 "$IPA_APP" > "$WORK/ipa-codesign-verify.log" 2>&1 || {
  cat "$WORK/ipa-codesign-verify.log" >&2
  fail "packaged IPA ad-hoc signature did not verify"
}
codesign -d --entitlements :- "$IPA_APP" > "$WORK/ipa-entitlements.plist" 2>"$WORK/ipa-codesign-display.log" \
  || fail "could not read entitlements from packaged IPA"
plutil -lint "$WORK/ipa-entitlements.plist" >/dev/null \
  || fail "packaged IPA entitlement blob is not a valid plist"
if privileged_entitlement_present "$WORK/ipa-entitlements.plist"; then
  fail "packaged IPA contains privileged entitlement: com.apple.developer.kernel.increased-memory-limit"
fi
ipa_archs="$(lipo -archs "$IPA_APP/$executable")"
[ "$ipa_archs" = "$archs" ] || fail "packaged IPA architectures changed: $ipa_archs (expected $archs)"

echo "AltStore-compatible source IPA: $OUTPUT"
echo "SHA-256: $(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
echo "Architectures: $archs"
echo "Verified from built app: CFBundleIdentifier=$bundle_id, MinimumOSVersion=$minimum_os"
echo "Verified packaged IPA entitlement blob: no privileged entitlements"
echo "NOTE: ad-hoc signature is integrity-checked only; it is NOT installable by iOS. AltStore/AltServer must re-sign with a free Apple ID."

#!/usr/bin/env bash
# Build an AltStore-source IPA from an already archived, Apple-unsigned app.
#
# This script never handles certificates, provisioning profiles, Apple ID
# passwords, or private keys. It applies an ad-hoc signature only so the
# repository entitlement blob is visible to AltStore before AltServer re-signs
# the app with a free Apple ID. The resulting IPA is NOT installable as-is.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTITLEMENTS="$ROOT/ios/LittleBitLoader/LittleBitLoader.entitlements"

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
for tool in codesign plutil lipo python3 shasum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: missing required tool: $tool" >&2; exit 1; }
done
[ -d "$APP" ] || { echo "ERROR: app bundle not found: $APP" >&2; exit 1; }
[ -f "$ENTITLEMENTS" ] || { echo "ERROR: entitlements file not found: $ENTITLEMENTS" >&2; exit 1; }

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

# Ad-hoc signature is a metadata carrier only. AltStore/AltServer replaces it
# with a free-Apple-ID signature at install time. No Apple identity is used here.
codesign --force --sign - --timestamp=none --entitlements "$ENTITLEMENTS" "$STAGED_APP" \
  || fail "ad-hoc signing with repository entitlements failed"

codesign -d --entitlements :- "$STAGED_APP" > "$WORK/app-entitlements.plist" 2>"$WORK/codesign-display.log" \
  || fail "could not read back the ad-hoc entitlement blob"
plutil -lint "$WORK/app-entitlements.plist" >/dev/null \
  || fail "entitlement blob readback is not a valid plist"
iml="$(plutil -extract com.apple.developer.kernel.increased-memory-limit raw -o - "$WORK/app-entitlements.plist" 2>/dev/null || true)"
[ "$iml" = "true" ] || fail "com.apple.developer.kernel.increased-memory-limit is missing from the ad-hoc signature"

mkdir -p "$(dirname "$OUTPUT")"
python3 "$ROOT/scripts/package_ipa.py" "$STAGED_APP" "$OUTPUT"

echo "AltStore-source IPA: $OUTPUT"
echo "SHA-256: $(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
echo "Architectures: $archs"
echo "Verified from built app: CFBundleIdentifier=$bundle_id, MinimumOSVersion=$minimum_os"
echo "Verified entitlement blob: com.apple.developer.kernel.increased-memory-limit=true"
echo "NOTE: ad-hoc signature is NOT installable by iOS. AltStore/AltServer must re-sign with a free Apple ID."

#!/usr/bin/env python3
import sys
import os
import zipfile
import shutil

def package_ipa(app_dir, output_ipa):
    if not os.path.isdir(app_dir):
        print(f"Error: {app_dir} is not a directory.")
        sys.exit(1)

    app_name = os.path.basename(os.path.normpath(app_dir))
    output_dir = os.path.dirname(os.path.abspath(output_ipa))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Packaging unsigned IPA from: {app_dir}")
    print(f"Output IPA path: {output_ipa}")

    with zipfile.ZipFile(output_ipa, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(app_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, app_dir)
                archive_path = os.path.join("Payload", app_name, rel_path)
                zipf.write(full_path, archive_path)

    file_size_mb = os.path.getsize(output_ipa) / (1024 * 1024)
    print(f"✓ Successfully created unsigned IPA: {output_ipa} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: package_ipa.py <path_to_app.app> [output.ipa]")
        sys.exit(1)

    app_path = sys.argv[1]
    out_ipa = sys.argv[2] if len(sys.argv) > 2 else "build/LittleBitLoader-unsigned.ipa"
    package_ipa(app_path, out_ipa)

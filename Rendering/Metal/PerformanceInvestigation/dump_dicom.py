#!/usr/bin/env python3
import os, sys, numpy as np, pydicom

# Directory containing the DICOM series (one file per slice, sorted by name).
SRC = os.environ.get("DICOM_SRC", "/Users/macair/Public/IMR/CTIMR/IMRToraceAddome/UZOZWT24/TQHNCPFG")
# Output raw u8 volume (matches the app's 512x512x1794 castToU8 texture).
OUT = os.environ.get("DICOM_OUT", "dicom.u8")

files = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)))
print("files:", len(files))

d0 = pydicom.dcmread(os.path.join(SRC, files[0]), force=True)
if not d0.file_meta.get('TransferSyntaxUID'):
    d0.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
print("ts:", d0.file_meta.TransferSyntaxUID)
print("rows:", d0.Rows, "cols:", d0.Columns, "bits:", d0.BitsAllocated,
      "signed:", getattr(d0, 'PixelRepresentation', None))
print("slope:", getattr(d0, 'RescaleSlope', 1), "intercept:", getattr(d0, 'RescaleIntercept', 0))

rows = int(d0.Rows); cols = int(d0.Columns)
n = len(files)
bits = int(d0.BitsAllocated)

out = np.zeros((n, rows, cols), dtype=np.uint8)
count = 0
for i, f in enumerate(files):
    d = pydicom.dcmread(os.path.join(SRC, f), force=True)
    if not d.file_meta.get('TransferSyntaxUID'):
        d.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    if not hasattr(d, 'Rows') or not hasattr(d, 'Columns') or not hasattr(d, 'PixelData'):
        continue
    arr = d.pixel_array  # handles slope/intercept automatically for CT
    # replicate harness: castToU8 shift 1024, scale 255/4095, clamp
    u8 = np.clip((arr.astype(np.float32) + 1024.0) * (255.0 / 4095.0), 0, 255).astype(np.uint8)
    out[count] = u8
    count += 1

out = out[:count]
out.tofile(OUT)
print("wrote", OUT, out.shape, out.nbytes)

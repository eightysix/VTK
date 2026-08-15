#!/usr/bin/env python3
import struct, os, sys

SRC = "/Users/macair/Public/IMR/CTIMR/IMRToraceAddome/UZOZWT24/TQHNCPFG"
OUT = "/var/folders/f2/wxp2wsd95dg_pkpzyxs8vqxr0000gn/T/opencode/probe/dicom.u8"

files = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)))
print("files:", len(files))

def read_file(p):
    d = open(p, 'rb').read()
    assert d[128:132] == b'DICM', p
    return d[132:]

def parse_header(d):
    """Return dict of {tag: value} for scalar tags we need. Handle explicit VR LE."""
    tags = {}
    i = 0
    n = len(d)
    pixel_data_offset = None
    while i + 8 <= n:
        g, e = struct.unpack_from('<HH', d, i)
        i += 4
        tag = (g << 16) | e
        # detect implicit vs explicit VR: if next 2 bytes are a valid VR
        vr = d[i:i+2]
        try:
            vr_str = vr.decode('ascii')
        except:
            vr_str = ''
        explicit = vr_str in ('AE','AS','AT','CS','DA','DS','DT','FL','FD','IS','LO','LT','OB','OD','OF','OL','OV','OW','PN','SH','SL','SQ','SS','ST','TM','UC','UI','UL','UN','UR','US','UT')
        if explicit:
            i += 2
            if vr_str in ('OB','OW','OF','OD','OL','OV','SQ','UN','UC','UR','UT'):
                if i + 8 > n: break
                len2 = struct.unpack_from('<I', d, i)[0]
                i += 4
                # skip 2 reserved bytes
                i += 2
                length = len2
            else:
                if i + 2 > n: break
                length = struct.unpack_from('<H', d, i)[0]
                i += 2
        else:
            # implicit VR, 4-byte length
            if i + 4 > n: break
            length = struct.unpack_from('<I', d, i)[0]
            i += 4
        if i + length > n:
            break
        val = d[i:i+length]
        if tag in (0x00020010,):  # transfer syntax
            tags[tag] = val.rstrip(b'\x00').decode('ascii')
        elif tag in (0x00280010, 0x00280011, 0x00280100, 0x00280102, 0x00280103, 0x00281050, 0x00281051):
            if length == 2:
                tags[tag] = struct.unpack('<H', val)[0]
            else:
                tags[tag] = val
        elif tag == 0x00281053:  # rescale slope, DS
            tags[tag] = val
        elif tag == 0x7FE00010:
            pixel_data_offset = i
        i += length
    return tags, pixel_data_offset, d

# First file: identify transfer syntax and geometry
first = files[0]
body = read_file(os.path.join(SRC, first))
tags, pdoff, d = parse_header(body)
print("transfer syntax:", tags.get(0x00020010))
print("rows:", tags.get(0x00280010), "cols:", tags.get(0x00280011))
print("bits:", tags.get(0x00280100), "signed:", tags.get(0x00280103))
print("pixel data offset:", pdoff, "body len:", len(body))

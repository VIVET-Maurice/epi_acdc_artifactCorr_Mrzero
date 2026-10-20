"""Write simulated chronological ADCs as native Siemens VD/VE Twix files.

The binary layout is produced by :mod:`twixtools`. All protocol buffers and
measurement headers are created here from the Pulseq acquisition description;
no patient header or acquired scanner sample is copied from a reference file.
The files are intended for software reconstruction tests, not scanner import.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import ctypes
import io
from pathlib import Path
import struct

import numpy as np


def _protocol_header(samples, kind, receiver_count):
    """Build binary protocol buffers, with Siemens header units made explicit."""
    nx, ny = map(int, samples["matrix_size"])
    fov = np.asarray(samples["FOV_m"], float)
    te = np.atleast_1d(samples["TE_s"]).astype(float)
    dwell = float(samples.get("dwell_s", np.median(np.diff(samples["elapsed_s"], axis=1))))
    if (fov.shape != (2,) or np.any(fov <= 0) or np.any(te <= 0)
            or not np.isfinite(dwell) or dwell <= 0):
        raise ValueError("Twix needs positive FOV, echo times and ADC dwell.")
    # ASCCONV fields use microseconds for TE/TR and nanoseconds for ADC dwell.
    # Float values preserve Pulseq's precision; the native reader accepts them.
    fields = {
        "tProtocolName": f'"MRZERO_SYNTHETIC_{kind.upper()}"',
        "sKSpace.lBaseResolution": nx,
        "sKSpace.ucDimension": 2,
        "sKSpace.lPhaseEncodingLines": ny,
        "sKSpace.lPartitions": 1,
        "sSliceArray.lSize": 1,
        "sSliceArray.asSlice[0].dReadoutFOV": fov[0] * 1000,
        "sSliceArray.asSlice[0].dPhaseFOV": fov[1] * 1000,
        "sSliceArray.asSlice[0].dThickness": float(samples.get("slice_thickness_mm", 5.0)),
        "sSliceArray.asSlice[0].sPosition.dSag": 0,
        "sSliceArray.asSlice[0].sPosition.dCor": 0,
        "sSliceArray.asSlice[0].sPosition.dTra": 0,
        "sSliceArray.asSlice[0].sNormal.dSag": 0,
        "sSliceArray.asSlice[0].sNormal.dCor": 0,
        "sSliceArray.asSlice[0].sNormal.dTra": 1,
        "sSliceArray.asSlice[0].dInPlaneRot": np.pi / 2,
        "alTR[0]": float(samples["TR_s"]) * 1e6,
        "adFlipAngleDegree[0]": 90.0,
        "sRXSPEC.alDwellTime[0]": dwell * 1e9,
        "sCoilSelectMeas.aRxCoilSelectData[0].asList[0].lElementSelected": 1,
        "lContrasts": len(te),
    }
    fields.update({f"alTE[{echo}]": float(value) * 1e6 for echo, value in enumerate(te)})
    lines = [f"{name} = {value}" for name, value in fields.items()]
    yaps = "### ASCCONV BEGIN ###\n" + "\n".join(lines) + "\n### ASCCONV END ###\n"
    # read_twix's default PMU parser looks up Dicom.SoftwareVersions even when
    # no physiological samples exist. This is a format label, not scanner data.
    dicom = '<ParamString."SoftwareVersions"> { "syngo MR E11" }\n'
    config = ('<ParamString."SequenceDescription"> { "MRZERO_SYNTHETIC" }\n'
              '<ParamString."tPatientName"> { "MRZERO_SYNTHETIC" }\n'
              '<ParamString."chronSliceIndices"> { "-" }\n'
              '<ParamLong."MeasUID"> { 1 }\n')
    meas = '<ParamString."tPatientPosition"> { "HFS" }\n'
    buffers = {"MeasYaps": yaps, "Dicom": dicom, "Config": config, "Meas": meas}
    payload = b"".join(name.encode("ascii") + b"\0" + struct.pack("<I", len(body.encode("ascii")))
                       + body.encode("ascii") for name, body in buffers.items())
    # Keep the first MDH aligned like a conventional VE measurement header.
    length = ((8 + len(payload) + 511) // 512) * 512
    raw = struct.pack("<II", length, len(buffers)) + payload
    raw += b"\0" * (length - len(raw))
    return np.frombuffer(raw, dtype="S1"), dwell


def _adc_counters(samples, kind):
    """Assign native row/echo counters without changing ADC acquisition order."""
    k = np.asarray(samples["k_mInv"], float)
    nx, ny = map(int, samples["matrix_size"])
    nav_count = int(samples["navigator_count"]) if kind == "scan" else 0
    echo_count = len(np.atleast_1d(samples["TE_s"])) if kind == "b0" else 1
    if k.ndim != 3 or k.shape[1:] != (nx, 3) or k.shape[0] != nav_count + ny * echo_count:
        raise ValueError("ADC trajectory does not describe the full requested Cartesian acquisition.")
    if nav_count not in (0, 3) or (kind != "scan" and nav_count):
        raise ValueError("Only zero navigators or the existing three-navigator EPI are supported.")
    ky = k[nav_count:, :, 1].mean(axis=1)
    row_coordinate = (ky - ky.min()) * float(np.asarray(samples["FOV_m"])[1])
    rows = np.rint(row_coordinate).astype(int)
    if (not np.allclose(row_coordinate, rows, rtol=0, atol=.002)
            or set(rows) != set(range(ny))):
        raise ValueError("ADC ky values do not cover unique Cartesian phase rows.")
    echoes = np.zeros(len(rows), dtype=int)
    if kind == "b0":
        histories = np.asarray(samples["rf_index"])[nav_count:]
        for history in np.unique(histories):
            indices = np.flatnonzero(histories == history)
            if len(indices) != echo_count or len(set(rows[indices])) != 1:
                raise ValueError("Each acquired B0 RF must have one row with every echo.")
            echoes[indices] = np.arange(echo_count)
    if len(set(zip(rows, echoes))) != ny * echo_count:
        raise ValueError("Duplicate Twix row/echo counters would overwrite an acquired ADC.")
    negative = np.mean(np.diff(k[..., 0], axis=1), axis=1) < 0
    if nav_count and negative[:nav_count].tolist() != [False, True, False]:
        raise ValueError("The three EPI navigators must have positive/negative/positive polarity.")
    return rows, echoes, negative, nav_count, echo_count


def write_synthetic_twix(path, signal, samples, kind):
    """Save native complex ADC samples and verify their binary round trip.

    ``signal`` has axes ``(receiver, ADC, readout)`` in acquisition chronology,
    including the three navigators when present. Negative readouts remain in
    acquisition order; their MDH ``REFLECT`` flag tells the reader to reverse
    them. ``kind`` is ``reference``, ``b0`` or ``scan``.

    Returns a small JSON-compatible validation report. Existing output files
    are rejected, matching ``twixtools.write_twix``'s exclusive creation mode.
    """
    import twixtools
    from twixtools import mdh_def
    from twixtools.mdb import Mdb_local

    if kind not in ("reference", "b0", "scan"):
        raise ValueError("kind must be reference, b0 or scan.")
    path = Path(path)
    if path.suffix.lower() != ".dat":
        raise ValueError("Native Twix output must use a .dat extension.")
    signal = np.asarray(signal)
    if (signal.ndim != 3 or signal.shape[0] < 1 or not np.iscomplexobj(signal)
            or not np.isfinite(signal).all()):
        raise ValueError("signal must be finite complex [receiver, ADC, readout] samples.")
    rows, echoes, negative, nav_count, echo_count = _adc_counters(samples, kind)
    nx, ny = map(int, samples["matrix_size"])
    if signal.shape[1:] != (len(negative), nx):
        raise ValueError("signal axes do not match the Pulseq ADC trajectory.")
    native_signal = np.asarray(signal, dtype=np.complex64)
    header, dwell = _protocol_header(samples, kind, signal.shape[0])
    blocks = []
    quaternion = np.asarray([1, 0, 0, 1], float) / np.sqrt(2)
    for adc in range(signal.shape[1]):
        block = Mdb_local(native_signal[:, adc, :])
        # twixtools 0.24 can alias its generated channel headers for N>1.
        # Use independent instances so ChannelId remains 0, 1, ..., N-1.
        block.channel_hdr = [mdh_def.Channel_header() for _ in range(signal.shape[0])]
        for channel, channel_header in enumerate(block.channel_hdr):
            channel_header.ChannelId = channel
            channel_header.MeasUID = 1
            channel_header.TypeAndChannelLength = ctypes.sizeof(channel_header) + 8 * nx
        block.mdh.MeasUID = 1
        block.mdh.CenterCol = nx // 2
        block.mdh.CenterLin = ny // 2
        block.mdh.SliceData.Quaternion[:] = quaternion
        block.add_flag("ONLINE")
        if negative[adc]:
            block.add_flag("REFLECT")
        if adc < nav_count:
            block.add_flag("PHASCOR")
        else:
            block.mdh.Counter.Lin = int(rows[adc - nav_count])
            block.mdh.Counter.Eco = int(echoes[adc - nav_count])
        if adc == 0:
            block.add_flag("FIRSTSCANINSLICE")
        if adc == signal.shape[1] - 1:
            block.add_flag("LASTSCANINSLICE")
            block.add_flag("LASTSCANINMEAS")
        blocks.append(block)
    # Supply ACQEND explicitly: its DMA includes the scan header and payload.
    # This avoids the writer's implicit ACQEND's inconsistent legacy length.
    end = Mdb_local()
    end.add_flag("ACQEND")
    end.mdh.MeasUID = 1
    mdh_def.set_dma_len(end.mdh, ctypes.sizeof(end.mdh) + len(end.data))
    blocks.append(end)
    path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        scan = {"hdr_str": header, "mdb": blocks,
                "hdr": {"Config": {"tPatientName": "MRZERO_SYNTHETIC",
                                   "SequenceDescription": f"MRZERO_SYNTHETIC_{kind.upper()}",
                                   "MeasUID": 1}}}
        twixtools.write_twix([scan], str(path), version_is_ve=True)
        decoded = twixtools.read_twix(str(path), verbose=False)[-1]
    yaps = decoded["hdr"]["MeasYaps"]
    slice_header = yaps["sSliceArray"]["asSlice"][0]
    expected_te = np.atleast_1d(samples["TE_s"])
    header_values = [
        (np.asarray(yaps["alTE"], dtype=float), expected_te * 1e6),
        (float(yaps["alTR"][0]), float(samples["TR_s"]) * 1e6),
        (yaps["sRXSPEC"]["alDwellTime"][0], dwell * 1e9),
        ([slice_header["dReadoutFOV"], slice_header["dPhaseFOV"]],
         np.asarray(samples["FOV_m"]) * 1000),
        (slice_header["dThickness"], float(samples.get("slice_thickness_mm", 5.0))),
        (yaps["adFlipAngleDegree"][0], 90.0),
    ]
    if any(not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1e-12, atol=1e-12)
           for actual, expected in header_values):
        raise RuntimeError("Twix round trip changed acquisition protocol metadata.")
    actual = decoded["mdb"]
    if len(actual) != native_signal.shape[1]:
        raise RuntimeError("Twix round trip changed the number of ADC blocks.")
    for adc, block in enumerate(actual):
        if not np.array_equal(block.data, native_signal[:, adc, :]):
            raise RuntimeError(f"Twix binary round trip changed ADC {adc}.")
        if (bool(block.is_flag_set("REFLECT")) != bool(negative[adc])
                or bool(block.is_flag_set("PHASCOR")) != bool(adc < nav_count)
                or block.mdh.CenterCol != nx // 2 or block.mdh.CenterLin != ny // 2):
            raise RuntimeError(f"Twix binary round trip changed ADC {adc} flags or center.")
        if adc >= nav_count and (block.cLin, block.cEco) != (int(rows[adc-nav_count]), int(echoes[adc-nav_count])):
            raise RuntimeError(f"Twix binary round trip changed ADC {adc} counters.")
        if [entry.ChannelId for entry in block.channel_hdr] != list(range(signal.shape[0])):
            raise RuntimeError("Twix binary round trip changed receiver channel IDs.")
    return {
        "format": "Siemens Twix VD/VE", "synthetic": True,
        "protocol_header_source": "generated from Pulseq; no acquired header copied",
        "binary_roundtrip_exact": True, "roundtrip_exact": True, "stored_dtype": "complex64",
        "image_shape": [ny, nx], "receiver_count": int(signal.shape[0]),
        "adc_count": int(signal.shape[1]), "echo_count": int(echo_count),
        "navigator_count": int(nav_count), "dwell_s": dwell,
        "ideal_excitation_flip_deg": 90.0, "file_size_bytes": path.stat().st_size,
    }


__all__ = ["write_synthetic_twix"]

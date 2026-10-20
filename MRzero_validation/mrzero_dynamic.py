"""MRzero proof of the single-RF, point-spin ACDC forward model.

This adapter uses native MRzero ``execute_graph(return_mag_adc=True)``. It
applies the accumulated DC-shim phase to each voxel's transverse magnetization
before receive summation. No MRzero package files are modified.

Scope: one ideal 90-degree pulse at t=0, no later RF, stationary point spins,
one uniform receive coil, and ideal instantaneous shim updates. T2 relaxation
is included; diffusion, T2-prime decay, and finite-voxel attenuation are disabled
to reproduce this repository's MATLAB ``forward_model.m``. This does not model
the RF slice profile, intravoxel dephasing, or the ACDC hardware response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def _field(obj: Any, name: str) -> Any:
    """Lit un champ d'un dictionnaire ou d'une structure MATLAB chargée par SciPy.

    `loadmat` peut fournir les paramètres sous forme de dictionnaire ou d'objet
    avec attributs. Cette fonction permet au reste du code d'accepter les deux.
    """
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def _map(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Convertit une valeur spatiale en tableau float64 de forme `shape`.

    Une valeur unique est répétée sur toute l'image ; un tableau déjà spatial
    doit avoir exactement la forme attendue.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.size == 1:
        return np.full(shape, float(array.reshape(-1)[0]))
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar or have image shape {shape}.")
    return array


def _prepare(image, b0_hz, shim_hz_per_amp, currents_a, params, seq):
    """Normalise et vérifie les entrées avant toute simulation.

    L'image est une densité de protons non négative, les cartes de shim ont la
    forme (Nx, Ny, Nc), et les courants (Nc, nombre d'événements). Les paramètres
    et la séquence peuvent être des dictionnaires ou des structures MATLAB lues
    avec SciPy. Dans `eventOfTime`, 0 signifie « shim éteint » et les valeurs
    positives sélectionnent une colonne de courants.

    Renvoie les données sous forme de tableaux NumPy normalisés. Une entrée
    incompatible provoque une ValueError explicite.
    """
    image = np.asarray(image)
    if np.iscomplexobj(image):
        if np.any(image.imag != 0):
            raise ValueError("image is proton density and must be real.")
        image = image.real
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("image must be a finite, nonnegative 2D PD array.")
    if image.sum() <= 0:
        raise ValueError("image must contain at least one nonzero voxel.")

    shape = image.shape
    b0 = _map(b0_hz, shape, "b0_hz")
    x = _map(_field(params, "X"), shape, "params.X")
    y = _map(_field(params, "Y"), shape, "params.Y")
    t1 = _map(_field(params, "T1"), shape, "params.T1")
    t2 = _map(_field(params, "T2"), shape, "params.T2")
    for name, array in (("B0", b0), ("X", x), ("Y", y)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite.")
    for name, array in (("T1", t1), ("T2", t2)):
        if np.isnan(array).any() or np.any(array <= 0):
            raise ValueError(f"{name} must be positive (infinity is allowed).")

    maps = np.asarray(shim_hz_per_amp, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[:, :, None]
    if maps.ndim != 3 or maps.shape[:2] != shape:
        raise ValueError("shim_hz_per_amp must have shape (Nx, Ny, Nc).")
    currents = np.asarray(currents_a, dtype=np.float64)
    if currents.ndim == 1:
        currents = currents.reshape(maps.shape[2], -1)
    if currents.ndim != 2 or currents.shape[0] != maps.shape[2]:
        raise ValueError("currents_a must have shape (Nc, number_of_events).")
    if not np.isfinite(maps).all() or not np.isfinite(currents).all():
        raise ValueError("Shim maps and currents must be finite.")

    gx = np.asarray(_field(seq, "Gx"), dtype=np.float64).reshape(-1)
    gy = np.asarray(_field(seq, "Gy"), dtype=np.float64).reshape(-1)
    adc_raw = np.asarray(_field(seq, "adc")).reshape(-1)
    events_raw = np.asarray(_field(seq, "eventOfTime"), dtype=np.float64).reshape(-1)
    dt_raw = np.asarray(_field(seq, "dt"), dtype=np.float64).reshape(-1)
    dt = np.full(gx.size, dt_raw[0]) if dt_raw.size == 1 else dt_raw
    if not (gx.size == gy.size == adc_raw.size == events_raw.size == dt.size):
        raise ValueError("Gx, Gy, adc, eventOfTime, and dt lengths must agree.")
    if not gx.size or not np.any(adc_raw):
        raise ValueError("The sequence must contain timesteps and ADC samples.")
    if not np.isin(adc_raw, [0, 1]).all():
        raise ValueError("adc must contain only 0/1 flags.")
    if not all(np.isfinite(a).all() for a in (gx, gy, events_raw, dt)):
        raise ValueError("Sequence arrays must be finite.")
    if np.any(dt <= 0):
        raise ValueError("Every timestep duration must be positive.")
    if np.any(events_raw != np.round(events_raw)):
        raise ValueError("eventOfTime must contain integer labels.")
    events = events_raw.astype(np.int64)
    if np.any(events < 0) or np.any(events > currents.shape[1]):
        raise ValueError("eventOfTime labels are 0=off, 1..number_of_events.")
    gamma_hz = float(np.asarray(_field(params, "gammaHz")).reshape(-1)[0])
    if not np.isfinite(gamma_hz) or gamma_hz <= 0:
        raise ValueError("params.gammaHz must be finite and positive, in Hz/T.")

    return dict(pd=image, b0=b0, maps=maps, currents=currents, x=x, y=y,
                t1=t1, t2=t2, gx=gx, gy=gy, adc=adc_raw.astype(bool),
                events=events, dt=dt, gamma_hz=gamma_hz)


def _analytical_prepared(p):
    """Calcule le signal ADC par récurrence NumPy, comme le modèle MATLAB.

    À chaque pas, additionne B0, le champ de la bobine active et celui des
    gradients, puis applique la rotation de phase et la décroissance T2 à chaque
    voxel. Le signal complexe est sommé uniquement aux instants ADC. Cette
    récurrence indépendante sert à vérifier le résultat de MRzero.
    """
    field_by_event = np.einsum("xyc,ce->xye", p["maps"], p["currents"])
    mxy = p["pd"].astype(np.complex128)
    signal = []
    for step in range(p["gx"].size):
        off_res_hz = p["b0"]
        event = p["events"][step]
        if p.get("time_currents") is not None:
            off_res_hz = off_res_hz + np.einsum(
                "xyc,c->xy", p["maps"], p["time_currents"][:, step]
            )
        elif event > 0:
            off_res_hz = off_res_hz + field_by_event[:, :, event - 1]
        omega_hz = off_res_hz + p["gamma_hz"] * (
            p["gx"][step] * p["x"] + p["gy"][step] * p["y"]
        )
        mxy *= np.exp(-2j * np.pi * omega_hz * p["dt"][step])
        mxy *= np.exp(-p["dt"][step] / p["t2"])
        if p["adc"][step]:
            signal.append(mxy.sum())
    return np.asarray(signal)


def analytical_adc_signal(image, b0_hz, shim_hz_per_amp, currents_a, params, seq,
                          *, currents_by_time_a=None):
    """Renvoie le signal ADC complexe calculé par le modèle NumPy indépendant.

    Prépare et valide les données, puis exécute la récurrence des spins ponctuels.
    `currents_by_time_a` peut fournir les courants moyens par canal et par pas ;
    sans cette forme d'onde, les courants sont choisis avec `eventOfTime`.
    """
    p = _prepare(image, b0_hz, shim_hz_per_amp, currents_a, params, seq)
    p["time_currents"] = _validate_time_currents(
        currents_by_time_a, p["maps"].shape[2], p["gx"].size
    )
    return _analytical_prepared(p)


def _validate_time_currents(currents_by_time_a, channels, timesteps):
    """Vérifie une forme d'onde de courants optionnelle, de forme (Nc, Nt).

    Renvoie `None` si aucune forme d'onde détaillée n'est fournie ; sinon,
    renvoie son tableau float64 après vérification de la forme et des valeurs.
    """
    if currents_by_time_a is None:
        return None
    waveform = np.asarray(currents_by_time_a, dtype=np.float64)
    if waveform.shape != (channels, timesteps) or not np.isfinite(waveform).all():
        raise ValueError("currents_by_time_a must be finite and have shape (Nc, Nt).")
    return waveform


def apply_dynamic_shim_phase(voxel_magnetization_adc, shim_hz_per_amp,
                              currents_a, seq, *, currents_by_time_a=None,
                              receive_sensitivity=None):
    """Applique la phase ACDC à chaque voxel avant de sommer le signal reçu.

    `voxel_magnetization_adc` a la forme (Nadc, Nx*Ny), avec les voxels aplatis
    dans l'ordre NumPy. Il contient déjà la densité de protons ainsi que les
    effets MRzero de B0, des gradients et de la phase native. Les cartes de shim
    sont en Hz/A.

    Le courant de chaque intervalle est intégré jusqu'aux instants ADC. La carte
    de shim transforme cette intégrale en cycles de phase par voxel ; la phase
    complexe est appliquée avant la somme, éventuellement pondérée par la
    sensibilité de réception. Sans sensibilité fournie, le poids vaut 1 partout.

    `currents_by_time_a`, s'il est fourni, contient les courants moyens par
    canal et par intervalle, de forme (Nc, Nt), et remplace les courants définis
    par événement. Le résultat est un signal complexe de longueur Nadc.
    """
    maps = np.asarray(shim_hz_per_amp, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[:, :, None]
    if maps.ndim != 3 or not np.isfinite(maps).all():
        raise ValueError("Shim maps must be finite with shape (Nx, Ny, Nc).")
    events_raw = np.asarray(_field(seq, "eventOfTime"), dtype=np.float64).reshape(-1)
    adc = np.asarray(_field(seq, "adc")).reshape(-1).astype(bool)
    dt_raw = np.asarray(_field(seq, "dt"), dtype=np.float64).reshape(-1)
    dt = np.full(events_raw.size, dt_raw[0]) if dt_raw.size == 1 else dt_raw
    if dt.size != events_raw.size or adc.size != events_raw.size:
        raise ValueError("adc, dt, and eventOfTime lengths must agree.")
    if not np.isfinite(events_raw).all() or np.any(events_raw != np.round(events_raw)):
        raise ValueError("eventOfTime must contain finite integer labels.")
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Timestep durations must be finite and positive.")
    events = events_raw.astype(np.int64)
    currents = np.asarray(currents_a, dtype=np.float64)
    if currents.ndim == 1:
        currents = currents.reshape(maps.shape[2], -1)
    if currents.ndim != 2 or currents.shape[0] != maps.shape[2]:
        raise ValueError("currents_a must have shape (Nc, Nevents).")
    if not np.isfinite(currents).all() or np.any(events < 0) or np.any(events > currents.shape[1]):
        raise ValueError("Invalid current values or event labels.")
    waveform = _validate_time_currents(currents_by_time_a, maps.shape[2], events.size)
    if waveform is None:
        per_step_current = np.zeros((events.size, maps.shape[2]))
        active = events > 0
        per_step_current[active] = currents[:, events[active] - 1].T
    else:
        per_step_current = waveform.T
    # Intégrer le courant jusqu'à chaque ADC : le produit courant × temps est en A·s.
    exposure_adc = np.cumsum(per_step_current * dt[:, None], axis=0)[adc]
    # Les cartes en Hz/A transforment cette intégrale en cycles de phase.
    shim_cycles = exposure_adc @ maps.reshape(-1, maps.shape[2]).T
    voxel_mag = np.asarray(voxel_magnetization_adc)
    if voxel_mag.shape != shim_cycles.shape:
        raise ValueError("ADC magnetization shape must be (Nadc, Nx*Ny).")
    receive = (np.ones(voxel_mag.shape[1]) if receive_sensitivity is None
               else np.asarray(receive_sensitivity).reshape(-1))
    if receive.size != voxel_mag.shape[1]:
        raise ValueError("Receive sensitivity must have one element per voxel.")
    return (voxel_mag * np.exp(-2j * np.pi * shim_cycles)) @ receive


def _relative_error(actual, expected):
    """Calcule l'erreur relative L2 entre deux signaux ou deux images.

    Le résultat est un ratio sans unité (0.01 correspond à 1 %). Le dénominateur
    est protégé contre le cas où la référence est nulle.
    """
    scale = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / scale)


def simulate_dynamic_acdc(image, b0_hz, shim_hz_per_amp, currents_a, params,
                          seq, *, compare_analytical=True,
                          currents_by_time_a=None,
                          return_native_magnetization=False):
    """Simule une lecture GRE-EPI avec un shim ACDC qui varie dans le temps.

    `image` est la densité de protons (PD), de forme (Nx, Ny). Les cartes de shim
    ont la forme (Nx, Ny, Nc), en Hz/A, et les courants la forme (Nc, Nevents),
    en A. `params` fournit notamment X/Y en m, T1/T2 en s et gammaHz en Hz/T.
    `seq` fournit Gx/Gy en T/m, dt, les instants ADC et `eventOfTime` : 0 signifie
    « shim éteint », les valeurs positives choisissent une colonne de courants
    avec la convention d'indices MATLAB. Les paramètres MATLAB chargés par SciPy
    peuvent être des dictionnaires ou des structures.

    Étapes principales :
    1. Construire une séquence MRzero à partir des durées, gradients et instants
       ADC de la séquence d'origine.
    2. Simuler B0, les gradients, T2 et l'aimantation de chaque voxel dans MRzero.
    3. Appliquer la phase ACDC accumulée à chaque voxel, puis sommer le signal.
    4. Calculer en option un signal de référence indépendant par récurrence NumPy.

    Renvoie un dictionnaire contenant `signal_adc` (MRzero avec ACDC),
    `static_signal_adc` (MRzero sans ACDC), `analytical_signal_adc` et les
    erreurs de comparaison. `compare_analytical=False` désactive la référence
    NumPy ; `return_native_magnetization=True` ajoute l'aimantation par voxel et
    la sensibilité de réception au résultat.

    Le modèle suppose un seul pulse RF idéal de 90°, des spins immobiles et
    ponctuels, une réception uniforme et des courants idéaux constants par
    intervalle. T2 est conservé ; diffusion, T2-prime et atténuation due au
    volume des voxels sont désactivés. La séquence MRzero est condensée aux
    instants ADC en préservant les durées et les moments intégrés des gradients.
    """
    import MRzeroCore as mr0
    import torch

    p = _prepare(image, b0_hz, shim_hz_per_amp, currents_a, params, seq)
    p["time_currents"] = _validate_time_currents(
        currents_by_time_a, p["maps"].shape[2], p["gx"].size
    )
    adc_steps = np.flatnonzero(p["adc"])
    sample_times = np.cumsum(p["dt"])[adc_steps]
    k_all = p["gamma_hz"] * np.cumsum(
        np.column_stack([p["gx"], p["gy"], np.zeros(p["gx"].size)])
        * p["dt"][:, None], axis=0
    )
    endpoints = np.unique(np.r_[adc_steps, p["gx"].size - 1])
    durations = np.diff(np.r_[0.0, np.cumsum(p["dt"])[endpoints]])
    moments = np.diff(np.vstack([np.zeros(3), k_all[endpoints]]), axis=0)

    rep = mr0.Repetition.zero(endpoints.size)
    rep.pulse.angle = torch.tensor(np.pi / 2, dtype=torch.float32)
    # MRzero emits -i exp(i RFphase); RFphase=pi/2 yields Mxy=PD.
    rep.pulse.phase = torch.tensor(np.pi / 2, dtype=torch.float32)
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.selective = False
    rep.event_time[:] = torch.from_numpy(durations.astype(np.float32))
    rep.gradm[:] = torch.from_numpy(moments.astype(np.float32))
    rep.adc_usage[:] = torch.from_numpy(p["adc"][endpoints].astype(np.int32))
    native_sequence = mr0.Sequence([rep], normalized_grads=False)

    coordinates = np.column_stack([
        p["x"].reshape(-1), p["y"].reshape(-1), np.zeros(p["pd"].size)
    ]).astype(np.float32)
    data = mr0.CustomVoxelPhantom(
        coordinates, PD=p["pd"].reshape(-1).astype(np.float32),
        B0=p["b0"].reshape(-1).astype(np.float32), B1=1.0,
        T1=p["t1"].reshape(-1).astype(np.float32),
        T2=p["t2"].reshape(-1).astype(np.float32), T2dash=np.inf,
        D=0.0, voxel_size=0.0, voxel_shape="box",
    ).build()
    # SimData clamps D to a positive minimum; explicitly disable diffusion.
    data.D = torch.zeros_like(data.D)
    data.dephasing_func = lambda k, _: torch.ones(k.shape[0], device=k.device)
    graph = mr0.compute_graph(native_sequence, data, max_state_count=16,
                              min_state_mag=0.0)
    with torch.no_grad():
        native_signal, magnetizations = mr0.execute_graph(
            graph, native_sequence, data, min_emitted_signal=0.0,
            min_latent_signal=0.0, print_progress=False, return_mag_adc=True,
        )
    if len(magnetizations) != 1 or len(magnetizations[0]) != 1:
        raise RuntimeError("Expected exactly one transverse coherence pathway.")
    # Returned voxel magnetization already includes PD and ADC phase.
    voxel_mag = magnetizations[0][0].detach().cpu().numpy()
    native_static = native_signal[:, 0].detach().cpu().numpy()
    receive = data.coil_sens[0].detach().cpu().numpy()
    static_resummed = voxel_mag @ receive

    dynamic_signal = apply_dynamic_shim_phase(
        voxel_mag, p["maps"], p["currents"], seq,
        currents_by_time_a=p["time_currents"], receive_sensitivity=receive,
    )
    analytical = _analytical_prepared(p) if compare_analytical else None

    result = dict(
        signal_adc=dynamic_signal, static_signal_adc=native_static,
        analytical_signal_adc=analytical,
        relative_error=(_relative_error(dynamic_signal, analytical)
                        if analytical is not None else None),
        max_abs_error=(float(np.max(np.abs(dynamic_signal - analytical)))
                       if analytical is not None else None),
        native_resummation_relative_error=_relative_error(static_resummed, native_static),
        sample_time_s=sample_times, k_adc_cycles_per_m=k_all[adc_steps],
        mrzero_version=mr0.__version__, native_event_count=int(endpoints.size),
    )
    if return_native_magnetization:
        result["voxel_magnetization_adc"] = voxel_mag
        result["receive_sensitivity"] = receive
    return result


def validate_synthetic(tolerance=2e-5):
    """Teste MRzero et NumPy sur de petits cas synthétiques contrôlés.

    Les cas couvrent le champ nul, B0 sans courant, un courant constant avec des
    périodes de shim éteint et des courants dynamiques de signes opposés. La
    fonction vérifie l'accord du signal MRzero avec la récurrence NumPy ainsi que
    la somme native de l'aimantation par voxel. Elle renvoie un rapport d'erreurs
    ou lève une AssertionError si `tolerance` est dépassée.
    """
    x, y = np.meshgrid(np.linspace(-0.031, 0.019, 3),
                       np.linspace(-0.023, 0.027, 4), indexing="ij")
    pd = np.array([[0.1, 0.3, 0.0, 0.7], [1.2, 0.2, 0.6, 0.4],
                   [0.9, 0.05, 0.8, 0.15]])
    b0 = 23 + 280 * x - 410 * y + 3100 * x * y
    maps = np.stack([35 + 700 * x, -18 + 450 * y + 1200 * x * y], axis=-1)
    params = dict(X=x, Y=y, T1=1.3, T2=0.045 + 0.12 * (x + 0.04),
                  gammaHz=42.57747892e6)
    seq = dict(Gx=0.004 * np.sin(np.arange(19) * 0.71),
               Gy=0.002 * np.cos(np.arange(19) * 0.43),
               dt=35e-6, adc=np.isin(np.arange(19), [2, 3, 5, 7, 10, 12, 15, 17]),
               eventOfTime=np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                     0, 3, 3, 3, 3, 4, 4, 4, 4]))
    varying = np.array([[0.4, -0.6, 0.2, 0.8], [-0.7, 0.3, 0.9, -0.2]])
    cases = {
        "zero_field": (np.zeros_like(b0), np.zeros_like(varying)),
        "nonzero_B0_zero_currents": (b0, np.zeros_like(varying)),
        "constant_currents_with_off_events": (b0, np.tile([[0.4], [-0.3]], (1, 4))),
        "dynamic_signed_currents": (b0, varying),
    }
    report = {}
    for name, (field, currents) in cases.items():
        result = simulate_dynamic_acdc(pd, field, maps, currents, params, seq)
        if result["relative_error"] >= tolerance:
            raise AssertionError(f"{name}: relative signal error {result['relative_error']}")
        if result["native_resummation_relative_error"] >= tolerance:
            raise AssertionError(f"{name}: native receive resummation mismatch")
        report[name] = {key: result[key] for key in (
            "relative_error", "max_abs_error", "native_resummation_relative_error",
            "mrzero_version", "native_event_count",
        )}
    return report


if __name__ == "__main__":
    print(json.dumps(validate_synthetic(), indent=2))

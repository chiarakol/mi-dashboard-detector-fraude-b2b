"""datos.py -- carga y preparación de los datos del panel.

Tres orígenes, todos terminan en el mismo formato:
  1. muestra()        datos inventados, para probar el panel sin conexión.
  2. cargar_delta()   lee las tablas Delta de tu Storage Account (contenedor "delta").
  3. cargar_csv()     lee los CSV que exporta verificar_corrida.py (o tablas Delta exportadas).

Todos devuelven un diccionario con cuatro DataFrames:
  tx        una fila por decisión (ya unida con la etiqueta real y la explicación)
  alertas   tabla alertas_drift
  semanal   serie semanal de deriva (solo en la muestra; con datos reales queda vacía)
  origen    texto que describe de dónde salieron los datos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UMBRALES = [0.3, 0.5, 0.7]
BRAZOS = ["0,3 · estricto", "0,5 · intermedio", "0,7 · laxo"]
DECISIONES = {"aprobar": "Aprobar", "revision_manual": "Revisión manual", "bloquear": "Bloquear"}

# Factores y pesos: los mismos de reglas_score.py (FACTORES)
FACTORES = {
    "monto_anomalo_vs_historico": ("Monto anómalo vs. historial", 0.15),
    "proveedor_poco_conectado": ("Proveedor poco conectado", 0.10),
    "forma_parte_de_ciclo_2a": ("Ciclo de pagos (tabular)", 0.10),
    "forma_parte_de_ciclo_2b": ("Ciclo de pagos (grafo)", 0.10),
    "es_relacion_nueva": ("Relación nueva", 0.05),
    "dispositivo_riesgo": ("Dispositivo de riesgo", 0.15),
    "fuera_horario": ("Fuera de horario", 0.05),
    "rafaga": ("Ráfaga de órdenes", 0.10),
    "documento_anomalo": ("Documento anómalo", 0.15),
    "cuenta_sospechosa": ("Cuenta sospechosa", 0.20),
    "atributos_compartidos": ("Atributos compartidos", 0.10),
    "paso_rapido": ("Paso rápido de fondos", 0.15),
    "proveedor_nuevo": ("Proveedor nuevo", 0.05),
}
TIPOS = {
    "toma_de_cuenta": "Toma de cuenta", "proveedor_fantasma": "Proveedor fantasma",
    "anillo_colusion": "Anillo de colusión", "fraccionamiento_ordenes": "Fraccionamiento de órdenes",
    "bust_out": "Bust-out", "factura_inflada_duplicada": "Factura inflada o duplicada",
    "bec_datos_bancarios": "BEC (datos bancarios)", "cadena_fondos_paso": "Cadena de fondos de paso",
}
_TIPO_FACT = {
    "toma_de_cuenta": ["dispositivo_riesgo", "fuera_horario", "monto_anomalo_vs_historico"],
    "proveedor_fantasma": ["proveedor_nuevo", "cuenta_sospechosa", "proveedor_poco_conectado"],
    "anillo_colusion": ["forma_parte_de_ciclo_2a", "forma_parte_de_ciclo_2b", "atributos_compartidos"],
    "fraccionamiento_ordenes": ["rafaga", "monto_anomalo_vs_historico"],
    "bust_out": ["monto_anomalo_vs_historico", "es_relacion_nueva", "rafaga"],
    "factura_inflada_duplicada": ["documento_anomalo", "monto_anomalo_vs_historico"],
    "bec_datos_bancarios": ["cuenta_sospechosa", "documento_anomalo", "dispositivo_riesgo"],
    "cadena_fondos_paso": ["paso_rapido", "forma_parte_de_ciclo_2b", "cuenta_sospechosa"],
}
TABLAS_DELTA = ["decisiones_cads", "decisiones_evaluadas", "alertas_drift", "explicaciones_shap_lite"]

# Matriz de recompensa: la misma de capa3b_aprendizaje_bandit.py
R_FRAUDE = {"bloquear": 1.0, "revision_manual": 0.5, "aprobar": -1.0}
R_LEGIT = {"bloquear": -1.0, "revision_manual": -0.25, "aprobar": 1.0}
ESCALA_MONTO, TOPE_MONTO, COSTO_FRICCION = 10000.0, 5.0, 0.05


def decidir(score: float, umbral: float) -> str:
    return "bloquear" if score >= umbral else ("revision_manual" if score >= umbral - 0.15 else "aprobar")


def recompensa(decision: str, fraude, monto) -> float:
    """Recompensa cost-sensitive de una decisión; NaN si aún no hay etiqueta real."""
    if fraude is None or pd.isna(fraude):
        return float("nan")
    valor = 1.0 if pd.isna(monto) else min(max(float(monto), 0.0) / ESCALA_MONTO, TOPE_MONTO)
    return R_FRAUDE[decision] * valor if bool(fraude) else R_LEGIT[decision] * COSTO_FRICCION


# ----------------------------------------------------------------------------------------------
# Preparación común
# ----------------------------------------------------------------------------------------------
def _a_bool(serie: pd.Series) -> pd.Series:
    """Convierte true/false/1/0 (texto o número) a booleano que admite vacíos."""
    m = serie.astype("string").str.strip().str.lower().map(
        {"true": True, "1": True, "verdadero": True, "false": False, "0": False, "falso": False})
    return m.astype("boolean")


def preparar(dec: pd.DataFrame, ev: pd.DataFrame | None = None, shap: pd.DataFrame | None = None) -> pd.DataFrame:
    tx = dec.copy()
    ts_col = "timestamp_evento" if "timestamp_evento" in tx else "timestamp_decision"
    tx["ts"] = pd.to_datetime(tx[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
    tx = tx.dropna(subset=["ts", "decision"])
    tx["dia"] = tx["ts"].dt.date
    tx["score"] = pd.to_numeric(tx["score_riesgo"], errors="coerce")
    tx["brazo"] = pd.to_numeric(tx.get("brazo"), errors="coerce").astype("Int64")
    um = pd.to_numeric(tx.get("umbral_usado"), errors="coerce")
    tx["umbral"] = um.fillna(tx["brazo"].map(lambda b: UMBRALES[int(b)] if pd.notna(b) else np.nan))
    tx["score_heur"] = pd.to_numeric(tx.get("score_heuristica"), errors="coerce")
    for col in ("monto", "comprador_id", "proveedor_id"):
        if col not in tx:
            tx[col] = np.nan
    tx["monto"] = pd.to_numeric(tx["monto"], errors="coerce")
    tx["fraude"] = pd.array([pd.NA] * len(tx), dtype="boolean")
    tx["tipo"] = pd.NA
    if ev is not None and len(ev):
        e = ev.drop_duplicates("event_id", keep="last").set_index("event_id")
        tx["fraude"] = tx["event_id"].map(_a_bool(e["es_fraude_real"])).astype("boolean")
        if "tipo_fraude_simulado" in e:
            tx["tipo"] = tx["event_id"].map(e["tipo_fraude_simulado"].replace("", np.nan))
    if shap is not None and len(shap):
        cols = [c for c in shap.columns if c.startswith("contrib_")]
        s = shap.drop_duplicates("event_id", keep="last").set_index("event_id")[cols]
        tx = tx.join(s, on="event_id")
    tx["recompensa"] = [recompensa(d, f, m) for d, f, m in zip(tx["decision"], tx["fraude"], tx["monto"])]
    return tx.sort_values("ts").reset_index(drop=True)


def _alertas(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["tipo_alerta", "metrica", "valor_baseline", "valor_actual", "detalle"]
    if df is None or not len(df):
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    for c in ("valor_baseline", "valor_actual"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[[c for c in df.columns if c in cols or c == "fecha"]].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 1. Datos de ejemplo
# ----------------------------------------------------------------------------------------------
def muestra(n: int = 1800, semilla: int = 2026) -> dict:
    rng = np.random.default_rng(semilla)
    claves = list(FACTORES)
    pesos = np.array([FACTORES[k][1] for k in claves])
    horas = np.repeat(np.arange(24), [1, 1, 1, 1, 1, 2, 4, 8, 14, 16, 16, 14, 12, 14, 16, 15, 12, 8, 5, 3, 2, 2, 1, 1])
    inicio = pd.Timestamp("2026-09-29")
    filas, ev, shap = [], [], []
    for i in range(n):
        dia = min(13, int(i / n * 14 + rng.random() * 0.99))
        ts = inicio + pd.Timedelta(days=dia, hours=int(rng.choice(horas)), seconds=int(rng.integers(0, 3600)))
        fraude = rng.random() < 0.04
        tipo = rng.choice(list(TIPOS)) if fraude else None
        if fraude:
            score = 0.28 + 0.65 * rng.random() ** 0.8
        elif rng.random() < 0.05:
            score = 0.3 + 0.3 * rng.random()
        else:
            score = rng.random() ** 3.2 * 0.5
        score = float(np.clip(score, 0.01, 0.99))
        heur = float(np.clip(score * 0.92 + rng.normal(0, 0.04), 0.01, 0.98))
        pref = _TIPO_FACT[tipo] if fraude else []
        act = np.array([(rng.random() < 0.9) if k in pref else (rng.random() < (0.22 if fraude else 0.12)) for k in claves])
        raw = pesos * np.where(act, 0.45 + 0.55 * rng.random(len(claves)), rng.random(len(claves)) * 0.06)
        contrib = raw / raw.sum() * heur
        p = np.array([.1, .8, .1]) if dia < 4 else np.array([.1 + .07 * (dia - 3), .8 - .06 * (dia - 3), .1 - .01 * (dia - 3)])
        brazo = int(rng.choice(3, p=p / p.sum()))
        monto = float(round(np.exp(np.log(3800) + rng.normal(0, 0.95)) * (2.2 if fraude else 1)))
        eid = f"ev-{100000 + i * 7 + int(rng.integers(0, 6)):x}"
        filas.append(dict(event_id=eid, timestamp_evento=ts, decision=decidir(score, UMBRALES[brazo]), brazo=brazo,
                          score_riesgo=score, umbral_usado=UMBRALES[brazo], score_heuristica=heur, monto=monto,
                          comprador_id=f"C-{int(rng.integers(1, 2001)):04d}", proveedor_id=f"P-{int(rng.integers(1, 601)):04d}",
                          modelo_version="v3"))
        if rng.random() < (0.97 if dia < 12 else 0.4):
            ev.append(dict(event_id=eid, es_fraude_real=bool(fraude), tipo_fraude_simulado=tipo))
        shap.append({"event_id": eid, **{f"contrib_{k}": float(v) for k, v in zip(claves, contrib)}})
    tx = preparar(pd.DataFrame(filas), pd.DataFrame(ev), pd.DataFrame(shap))
    semanal = pd.DataFrame({
        "semana": pd.to_datetime(["2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05", "2026-10-12"]),
        "psi_score": [.03, .04, .05, .04, .08, .12, .31, .27], "psi_brazos": [.02, .03, .03, .06, .09, .14, .19, .28],
        "recall": [.71, .70, .72, .69, .68, .66, .58, .52]})
    alertas = pd.DataFrame([
        dict(tipo_alerta="drift_distribucion", metrica="score_riesgo", valor_baseline=np.nan, valor_actual=.31, detalle="PSI=0.310 > umbral 0.25 (semana del 05/10)"),
        dict(tipo_alerta="drift_distribucion", metrica="score_riesgo", valor_baseline=np.nan, valor_actual=.27, detalle="PSI=0.270 > umbral 0.25 (semana del 12/10)"),
        dict(tipo_alerta="drift_distribucion", metrica="brazo_bandit", valor_baseline=np.nan, valor_actual=.28, detalle="PSI=0.280 > umbral 0.25, el bandit cambió de umbral"),
        dict(tipo_alerta="drift_desempeno", metrica="recall_fraude", valor_baseline=.71, valor_actual=.52, detalle="El recall cayó 0.19 (límite 0.15)")])
    return dict(tx=tx, alertas=alertas, semanal=semanal, origen="Datos de ejemplo (inventados)", real=False)


# ----------------------------------------------------------------------------------------------
# 2. Azure Storage (tablas Delta)
# ----------------------------------------------------------------------------------------------
def cargar_delta(cuenta: str, clave: str, entorno: str = "tesis", contenedor: str = "delta") -> dict:
    from deltalake import DeltaTable  # import tardío: solo hace falta en este modo
    opciones = {"account_name": cuenta, "account_key": clave}
    tablas, faltan = {}, []
    for t in TABLAS_DELTA:
        uri = f"abfss://{contenedor}@{cuenta}.dfs.core.windows.net/{entorno}/{t}"
        try:
            tablas[t] = DeltaTable(uri, storage_options=opciones).to_pandas()
        except Exception as e:  # tabla aún no creada o sin permiso
            tablas[t] = None
            faltan.append(t)
            if t == "decisiones_cads":
                raise RuntimeError(f"No se pudo leer decisiones_cads en {uri}: {e}") from e
    tx = preparar(tablas["decisiones_cads"], tablas["decisiones_evaluadas"], tablas["explicaciones_shap_lite"])
    nota = f" (sin: {', '.join(faltan)})" if faltan else ""
    return dict(tx=tx, alertas=_alertas(tablas["alertas_drift"]), semanal=pd.DataFrame(),
                origen=f"Azure · {cuenta} · entorno {entorno}{nota}", real=True)


# ----------------------------------------------------------------------------------------------
# 3. CSV
# ----------------------------------------------------------------------------------------------
def cargar_csv(archivos) -> dict:
    """`archivos`: lista de objetos con .name y legibles por pandas (los de st.file_uploader)."""
    dec = ev = shap = al = None
    usados = []
    for f in archivos:
        df = pd.read_csv(f)
        c = set(df.columns)
        if "tipo_alerta" in c:
            al = df
        elif "es_fraude_real" in c:
            ev = df
        elif "shap_top" in c or any(x.startswith("contrib_") for x in c):
            shap = df
        elif {"brazo", "score_riesgo"} <= c:
            dec = df
        else:
            continue
        usados.append(f.name)
    if dec is None:
        raise ValueError("Falta decisiones_cads.csv (debe tener las columnas brazo y score_riesgo).")
    return dict(tx=preparar(dec, ev, shap), alertas=_alertas(al), semanal=pd.DataFrame(),
                origen="CSV: " + ", ".join(usados), real=True)

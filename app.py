"""Panel de Riesgo B2B (Streamlit).

Ejecutar:   streamlit run app.py
Datos:      ejemplo (por defecto), Azure Storage (Delta) o CSV. Ver README.md.
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import datos as D

st.set_page_config(page_title="Panel de Riesgo B2B", page_icon="🛡️", layout="wide")

VERDE, AMBAR, ROJO = "#1a7f4b", "#d98a00", "#c8372d"
COLOR_DEC = alt.Scale(domain=["Aprobar", "Revisión manual", "Bloquear"], range=[VERDE, AMBAR, ROJO])
COLOR_BRAZO = alt.Scale(domain=D.BRAZOS, range=["#2a78d6", "#eb6834", "#1baf7a"])
COLOR_REAL = alt.Scale(domain=["Legítima confirmada", "Fraude confirmado", "Sin etiqueta aún"], range=["#2a78d6", "#eb6834", "#b8c2cc"])
GLIFO = {"aprobar": "● Aprobar", "revision_manual": "▲ Revisión manual", "bloquear": "■ Bloquear"}

st.markdown("""<style>
div[data-testid="stMetricValue"] {font-family: "IBM Plex Mono", ui-monospace, monospace;}
.block-container {padding-top: 2rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ carga de datos
def _secreto(nombre: str, defecto: str = "") -> str:
    try:
        if nombre in st.secrets:
            return str(st.secrets[nombre])
    except Exception:
        pass
    return os.environ.get(nombre, defecto)


@st.cache_data(show_spinner="Leyendo tablas Delta de Azure…", ttl=300)
def _delta(cuenta: str, clave: str, entorno: str):
    return D.cargar_delta(cuenta, clave, entorno)


@st.cache_data(show_spinner=False)
def _muestra():
    return D.muestra()


def obtener_datos() -> dict:
    st.sidebar.title("Panel de Riesgo B2B")
    st.sidebar.caption("Detección de fraude entre empresas")
    cuenta, clave = _secreto("AZURE_STORAGE_ACCOUNT"), _secreto("AZURE_STORAGE_KEY")
    opciones = ["Datos de ejemplo", "Subir CSV"] + (["Azure (Delta)"] if cuenta and clave else [])
    modo = st.sidebar.radio("Origen de los datos", opciones, index=len(opciones) - 1 if len(opciones) == 3 else 0)
    try:
        if modo == "Azure (Delta)":
            entorno = st.sidebar.selectbox("Entorno", ["tesis", "test"], index=["tesis", "test"].index(_secreto("PIPELINE_ENTORNO", "tesis")) if _secreto("PIPELINE_ENTORNO", "tesis") in ("tesis", "test") else 0)
            if st.sidebar.button("Actualizar ahora"):
                _delta.clear()
            return _delta(cuenta, clave, entorno)
        if modo == "Subir CSV":
            archivos = st.sidebar.file_uploader("CSV de la corrida", type="csv", accept_multiple_files=True,
                                                help="decisiones_cads.csv es obligatorio. Opcionales: decisiones_evaluadas, explicaciones_shap_lite y alertas_drift.")
            if archivos:
                return D.cargar_csv(archivos)
            st.sidebar.info("Sube al menos decisiones_cads.csv. Mientras tanto se muestran datos de ejemplo.")
    except Exception as e:
        st.sidebar.error(f"No se pudieron cargar los datos: {e}")
    return _muestra()


# ------------------------------------------------------------------ utilidades de gráficos
def n(x):
    return f"{int(x):,}".replace(",", ".")


def p(x, d=0):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def f2(x):
    return "—" if pd.isna(x) else f"{x:.2f}".replace(".", ",")


def con_etiqueta(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.copy()
    t["Decisión"] = t["decision"].map(D.DECISIONES)
    t["Realidad"] = np.where(t["fraude"].isna(), "Sin etiqueta aún", np.where(t["fraude"].fillna(False).astype(bool), "Fraude confirmado", "Legítima confirmada"))
    return t


def tabla_transacciones(df: pd.DataFrame, clave: str):
    """Muestra una tabla seleccionable y devuelve la fila elegida (o None)."""
    v = pd.DataFrame({
        "Evento": df["event_id"], "Fecha": df["ts"], "Decisión": df["decision"].map(GLIFO), "Score": df["score"],
        "Umbral": df["umbral"], "Monto": df["monto"], "Comprador": df["comprador_id"], "Proveedor": df["proveedor_id"],
        "Realidad": con_etiqueta(df)["Realidad"]})
    ev = st.dataframe(v, hide_index=True, width="stretch", height=min(40 + 35 * len(v), 420), key=clave,
                      on_select="rerun", selection_mode="single-row",
                      column_config={"Fecha": st.column_config.DatetimeColumn(format="DD/MM HH:mm"),
                                     "Score": st.column_config.NumberColumn(format="%.2f"),
                                     "Umbral": st.column_config.NumberColumn(format="%.1f"),
                                     "Monto": st.column_config.NumberColumn(format="%,.0f")})
    filas = ev.selection.rows
    return df.iloc[filas[0]] if filas else None


def detalle(fila):
    st.divider()
    st.subheader(f"Detalle de {fila['event_id']}")
    c = st.columns(5)
    c[0].metric("Decisión", D.DECISIONES[fila["decision"]])
    c[1].metric("Score de riesgo", f2(fila["score"]))
    c[2].metric("Umbral usado", "—" if pd.isna(fila["umbral"]) else f"{fila['umbral']:.1f}".replace(".", ","))
    c[3].metric("Monto", "—" if pd.isna(fila["monto"]) else n(fila["monto"]))
    real = "Sin etiqueta aún" if pd.isna(fila["fraude"]) else ("Fraude · " + D.TIPOS.get(fila["tipo"], str(fila["tipo"] or "")) if fila["fraude"] else "Legítima")
    c[4].metric("Resultado real", real)
    if not pd.isna(fila["recompensa"]):
        st.caption(f"Recompensa del bandit para esta decisión: {fila['recompensa']:+.3f}".replace(".", ","))
    cols = [x for x in fila.index if x.startswith("contrib_") and pd.notna(fila[x])]
    st.markdown("**¿Por qué este score?** Cuánto aportó cada factor (descomposición aditiva del score heurístico, con los pesos del CADS).")
    if not cols:
        st.info("No hay explicación para este evento. Carga explicaciones_shap_lite para verla.")
        return
    e = pd.DataFrame({"Factor": [D.FACTORES.get(x[8:], (x[8:],))[0] for x in cols], "Aporte": [float(fila[x]) for x in cols]})
    e = e[e["Aporte"] > 0.004].sort_values("Aporte", ascending=False).head(8)
    ch = alt.Chart(e).mark_bar(color="#17507f", cornerRadiusEnd=3, size=16).encode(
        x=alt.X("Aporte:Q", title="Aporte al score"), y=alt.Y("Factor:N", sort="-x", title=None),
        tooltip=["Factor", alt.Tooltip("Aporte:Q", format=".3f")]).properties(height=max(120, 30 * len(e)))
    st.altair_chart(ch, width="stretch")
    st.caption("Esta explicación permite que una persona entienda el motivo de la decisión, como pide la Ley N.° 31814 para sistemas de IA.")


# ------------------------------------------------------------------ vistas
def vista_resumen(tx, alertas):
    st.header("Resumen")
    st.caption("Qué decidió el sistema sobre las transacciones recientes y qué quedó para revisión.")
    if len(alertas):
        ult = alertas.iloc[-1]
        st.error(f"**Alerta de deriva activa.** {ult['detalle']} Revisa la pestaña Deriva.", icon="⚠️")
    else:
        st.success("**Sin alertas de deriva.** El comportamiento reciente coincide con la línea base.", icon="✅")
    t = con_etiqueta(tx)
    fr = t[t["Realidad"] == "Fraude confirmado"]
    recall = (fr["decision"] != "aprobar").mean() if len(fr) else np.nan
    c = st.columns(4)
    c[0].metric("Transacciones decididas", n(len(t)), f"{t['ts'].min():%d/%m} al {t['ts'].max():%d/%m}", delta_color="off", delta_arrow="off")
    c[1].metric("Bloqueadas", n((t["decision"] == "bloquear").sum()), p((t["decision"] == "bloquear").mean(), 1) + " del total", delta_color="off", delta_arrow="off")
    c[2].metric("En revisión manual", n((t["decision"] == "revision_manual").sum()), p((t["decision"] == "revision_manual").mean(), 1) + " del total", delta_color="off", delta_arrow="off")
    c[3].metric("Fraude capturado", "—" if pd.isna(recall) else p(recall), f"{len(fr)} fraudes confirmados", delta_color="off", delta_arrow="off")

    a, b = st.columns(2)
    with a:
        st.markdown("**Alertas por día** · revisión manual y bloqueos")
        d = t[t["decision"] != "aprobar"].groupby(["dia", "Decisión"]).size().reset_index(name="Transacciones")
        d["dia"] = pd.to_datetime(d["dia"])
        st.altair_chart(alt.Chart(d).mark_bar().encode(
            x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m")), y=alt.Y("Transacciones:Q", title=None),
            color=alt.Color("Decisión:N", scale=alt.Scale(domain=["Revisión manual", "Bloquear"], range=[AMBAR, ROJO]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("dia:T", format="%d/%m"), "Decisión", "Transacciones"]).properties(height=260), width="stretch")
    with b:
        bajo = int((t["score"] < .1).sum())
        st.markdown(f"**Distribución del score** · desde 0,10 ({n(bajo)} transacciones, {p(bajo / len(t))}, quedan por debajo)")
        h = t[t["score"] >= .1]
        barras = alt.Chart(h).mark_bar().encode(
            x=alt.X("score:Q", bin=alt.Bin(step=0.05, extent=[0.1, 1.0]), title="Score de riesgo"), y=alt.Y("count():Q", title=None),
            color=alt.Color("Realidad:N", scale=COLOR_REAL, legend=alt.Legend(orient="top", title=None)))
        reglas = alt.Chart(pd.DataFrame({"u": D.UMBRALES})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="u:Q")
        st.altair_chart((barras + reglas).properties(height=260), width="stretch")

    st.markdown("**Cola de revisión manual** · selecciona una fila para ver por qué tiene ese score")
    cola = tx[tx["decision"] == "revision_manual"].sort_values("ts", ascending=False).head(10)
    if cola.empty:
        st.info("No hay transacciones en revisión manual.")
        return
    fila = tabla_transacciones(cola, "tabla_cola")
    if fila is not None:
        detalle(fila)


def vista_transacciones(tx):
    st.header("Transacciones")
    st.caption("Todas las decisiones del CADS. Selecciona una fila para ver el detalle y la explicación del score.")
    t = con_etiqueta(tx)
    c = st.columns([2, 1, 1, 1])
    q = c[0].text_input("Buscar", placeholder="Evento, comprador o proveedor")
    dec = c[1].selectbox("Decisión", ["Todas"] + list(D.DECISIONES.values()))
    bra = c[2].selectbox("Umbral elegido", ["Todos"] + D.BRAZOS)
    rea = c[3].selectbox("Resultado real", ["Todos", "Fraude confirmado", "Legítima confirmada", "Sin etiqueta aún"])
    if q:
        m = t["event_id"].str.contains(q, case=False, na=False) | t["comprador_id"].astype("string").str.contains(q, case=False, na=False) | t["proveedor_id"].astype("string").str.contains(q, case=False, na=False)
        t = t[m]
    if dec != "Todas":
        t = t[t["Decisión"] == dec]
    if bra != "Todos":
        t = t[t["brazo"] == D.BRAZOS.index(bra)]
    if rea != "Todos":
        t = t[t["Realidad"] == rea]
    t = t.sort_values("ts", ascending=False)
    st.caption(f"{n(len(t))} transacciones (se muestran las 500 más recientes)")
    if t.empty:
        st.info("Ninguna transacción coincide con los filtros.")
        return
    fila = tabla_transacciones(t.head(500), "tabla_tx")
    if fila is not None:
        detalle(fila)


def vista_aprendizaje(tx):
    st.header("Aprendizaje del bandit")
    st.caption("El sistema elige entre tres umbrales y aprende de las etiquetas reales. Dejar pasar un fraude pesa más que bloquear a un cliente legítimo.")
    t = con_etiqueta(tx)
    ev = t[t["fraude"].notna()]
    fr = ev[ev["fraude"].fillna(False).astype(bool)]
    por_brazo = t["brazo"].value_counts()
    top = int(por_brazo.idxmax()) if len(por_brazo) else 1
    c = st.columns(4)
    c[0].metric("Decisiones evaluadas", n(len(ev)), p(len(ev) / len(t)) + " ya tiene etiqueta", delta_color="off", delta_arrow="off")
    c[1].metric("Umbral más usado", f"{D.UMBRALES[top]:.1f}".replace(".", ","), D.BRAZOS[top].split(" · ")[1], delta_color="off", delta_arrow="off")
    c[2].metric("Recompensa acumulada", f2(ev["recompensa"].sum()).replace(",", ",") if len(ev) else "—")
    c[3].metric("Fraude bloqueado directo", "—" if fr.empty else p((fr["decision"] == "bloquear").mean()), "sin contar revisión manual", delta_color="off", delta_arrow="off")

    a, b = st.columns(2)
    with a:
        st.markdown("**Umbral elegido por día** · proporción de decisiones con cada umbral")
        d = t.dropna(subset=["brazo"]).assign(Umbral=lambda x: x["brazo"].astype(int).map(lambda i: D.BRAZOS[i])).groupby(["dia", "Umbral"]).size().reset_index(name="n")
        d["dia"] = pd.to_datetime(d["dia"])
        st.altair_chart(alt.Chart(d).mark_bar().encode(
            x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m")), y=alt.Y("n:Q", stack="normalize", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("Umbral:N", scale=COLOR_BRAZO, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("dia:T", format="%d/%m"), "Umbral", alt.Tooltip("n:Q", title="Decisiones")]).properties(height=260), width="stretch")
    with b:
        st.markdown("**Recompensa acumulada** · sobre decisiones con etiqueta real")
        r = ev.groupby("dia")["recompensa"].sum().cumsum().reset_index()
        r["dia"] = pd.to_datetime(r["dia"])
        line = alt.Chart(r).mark_line(point=alt.OverlayMarkDef(filled=True, size=50), color="#2a78d6").encode(
            x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m")), y=alt.Y("recompensa:Q", title=None),
            tooltip=[alt.Tooltip("dia:T", format="%d/%m"), alt.Tooltip("recompensa:Q", format=".2f", title="Acumulada")])
        st.altair_chart(line.properties(height=260), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Fraude capturado por tipología** · revisión o bloqueo sobre fraudes confirmados")
        if fr.empty or fr["tipo"].isna().all():
            st.info("Aún no hay fraudes con tipología.")
        else:
            g = fr.groupby("tipo").agg(fraudes=("decision", "size"), capturados=("decision", lambda s: (s != "aprobar").sum())).reset_index()
            g["Tipología"] = g["tipo"].map(lambda k: D.TIPOS.get(k, k))
            g["Capturado"] = g["capturados"] / g["fraudes"]
            g["Detalle"] = g["capturados"].astype(str) + " de " + g["fraudes"].astype(str)
            st.altair_chart(alt.Chart(g).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Capturado:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                y=alt.Y("Tipología:N", sort="-x", title=None),
                color=alt.condition(alt.datum.Capturado < 0.6, alt.value(ROJO), alt.value("#17507f")),
                tooltip=["Tipología", "Detalle", alt.Tooltip("Capturado:Q", format=".0%")]).properties(height=max(180, 32 * len(g))), width="stretch")
    with b:
        st.markdown("**Decisión frente a la realidad** · el fraude aprobado es el error más caro")
        if ev.empty:
            st.info("Aún no hay decisiones con etiqueta.")
        else:
            m = pd.crosstab(ev["Decisión"], ev["Realidad"]).reindex(["Aprobar", "Revisión manual", "Bloquear"]).fillna(0).astype(int)
            for col in ("Legítima confirmada", "Fraude confirmado"):
                if col not in m:
                    m[col] = 0
            st.dataframe(m[["Legítima confirmada", "Fraude confirmado"]], width="stretch")
            aprobados = int(m.loc["Aprobar", "Fraude confirmado"])
            if aprobados:
                st.warning(f"{aprobados} fraudes fueron aprobados.")


def vista_deriva(alertas, semanal):
    st.header("Deriva del modelo (DTRO)")
    st.caption("Cada lunes el sistema compara su comportamiento reciente con una línea base. Las alertas quedan registradas; no se reentrena solo.")
    if len(alertas):
        st.error(f"**{len(alertas)} alertas registradas.** La última: {alertas.iloc[-1]['detalle']}", icon="⚠️")
    else:
        st.success("**Sin alertas.** No hay deriva registrada.", icon="✅")
    if len(semanal):
        a, b = st.columns(2)
        with a:
            st.markdown("**Cambio en la distribución (PSI)** · sobre 0,25 se registra una alerta")
            m = semanal.melt("semana", ["psi_score", "psi_brazos"], var_name="Serie", value_name="PSI")
            m["Serie"] = m["Serie"].map({"psi_score": "Score de riesgo", "psi_brazos": "Brazos elegidos"})
            l = alt.Chart(m).mark_line(point=True).encode(x=alt.X("semana:T", title=None, axis=alt.Axis(format="%d/%m")), y=alt.Y("PSI:Q", title=None),
                color=alt.Color("Serie:N", scale=alt.Scale(range=["#2a78d6", "#eb6834"]), legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("semana:T", format="%d/%m"), "Serie", alt.Tooltip("PSI:Q", format=".2f")])
            r = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(color=ROJO, strokeDash=[5, 4]).encode(y="y:Q")
            st.altair_chart((l + r).properties(height=260), width="stretch")
        with b:
            base = 0.71
            st.markdown(f"**Recall de fraude** · línea base {base:.2f}, se alerta si cae más de 0,15".replace(".", ","))
            l = alt.Chart(semanal).mark_line(point=True, color="#2a78d6").encode(x=alt.X("semana:T", title=None, axis=alt.Axis(format="%d/%m")),
                y=alt.Y("recall:Q", title=None, scale=alt.Scale(domain=[0.4, 0.8])), tooltip=[alt.Tooltip("semana:T", format="%d/%m"), alt.Tooltip("recall:Q", format=".2f")])
            r = alt.Chart(pd.DataFrame({"y": [base - 0.15]})).mark_rule(color=ROJO, strokeDash=[5, 4]).encode(y="y:Q")
            st.altair_chart((l + r).properties(height=260), width="stretch")
    else:
        st.info("Con datos reales no hay serie semanal guardada: DTRO solo registra las alertas. Aquí abajo está su historial.")
    st.markdown("**Historial de alertas**")
    if len(alertas):
        v = alertas.rename(columns={"tipo_alerta": "Tipo", "metrica": "Métrica", "valor_baseline": "Línea base", "valor_actual": "Valor actual", "detalle": "Detalle"})
        st.dataframe(v, hide_index=True, width="stretch")
    else:
        st.caption("No hay alertas. Si usas CSV, sube alertas_drift.csv para verlas aquí.")


# ------------------------------------------------------------------ principal
datos = obtener_datos()
tx = datos["tx"]
st.sidebar.markdown(f"{'🟢' if datos['real'] else '🟡'} **{datos['origen']}**")
if tx.empty:
    st.warning("No hay decisiones para mostrar.")
    st.stop()
tabs = st.tabs(["Resumen", "Transacciones", "Aprendizaje", f"Deriva ({len(datos['alertas'])})" if len(datos["alertas"]) else "Deriva"])
with tabs[0]:
    vista_resumen(tx, datos["alertas"])
with tabs[1]:
    vista_transacciones(tx)
with tabs[2]:
    vista_aprendizaje(tx)
with tabs[3]:
    vista_deriva(datos["alertas"], datos["semanal"])

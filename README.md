# 📈 Analista Técnico

Dashboard de análisis técnico estilo *price-action* con detección automática de:
- **Swing highs / lows**
- **Soportes y resistencias** (clusterizados por toques)
- **Trendlines** dinámicas
- **Canales** paralelos
- **Retrocesos de Fibonacci**
- **Patrones**: Doble Techo, Doble Piso, Triángulos (simétrico, ascendente, descendente)

Datos en tiempo real (con delay) vía Yahoo Finance — funciona con acciones US, cripto, ADRs argentinos, BCBA e índices.

## 🚀 Cómo correrlo

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar el dashboard
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## 🎯 Tickers soportados

| Tipo | Ejemplos |
|---|---|
| Acciones US | `AAPL`, `TSLA`, `NVDA`, `MSFT`, `META` |
| Cripto | `BTC-USD`, `ETH-USD`, `SOL-USD` |
| ADRs argentinos (NYSE) | `YPF`, `GGAL`, `PAM`, `BMA`, `TEO`, `MELI` |
| BCBA (Bs As) | `YPFD.BA`, `GGAL.BA`, `ALUA.BA`, `PAMP.BA` |
| Índices | `^GSPC` (S&P 500), `^IXIC` (Nasdaq), `^MERV` (Merval) |
| Forex | `EURUSD=X`, `USDARS=X` |

## ⚙️ Cómo afinar las detecciones

En el sidebar:
- **Distancia mínima entre swings**: subila si te aparecen demasiados swings chiquitos.
- **Prominencia mínima**: porcentaje del precio que un movimiento debe tener para contar como swing.

Si no aparecen patrones, suele ser por una de tres cosas:
1. Período muy corto → ampliá a 1 año o más.
2. Sensibilidad muy alta → bajá distancia o prominencia.
3. Genuinamente no hay un patrón clásico ahora mismo.

## 📁 Estructura

```
analista_tecnico/
├── app.py            # Dashboard Streamlit
├── analysis.py       # Lógica pura de detección (sin UI)
├── requirements.txt
└── README.md
```

`analysis.py` es independiente del UI: podés importarlo desde un notebook,
un script de alertas, o lo que se te ocurra.

## ⚠️ Disclaimer

Esto **no es asesoramiento financiero**. Las detecciones automáticas son un
punto de partida — la lectura de un gráfico sigue siendo un ejercicio
discrecional. Usalo como apoyo, no como oráculo.

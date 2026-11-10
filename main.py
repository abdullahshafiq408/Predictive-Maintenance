import pickle
import sqlite3
import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import cm

base_dir = Path(__file__).parent
models_dir = base_dir / "models"
db_path = base_dir / "predictions.db"

# load models
with open(models_dir / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(models_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(models_dir / "encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open(models_dir / "features.pkl", "rb") as f:
    features = pickle.load(f)

type_map = {label: int(encoder.transform([label])[0])
            for label in encoder.classes_}


def init_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                machine_type TEXT,
                air_temp REAL,
                process_temp REAL,
                rotational_speed INTEGER,
                torque REAL,
                tool_wear INTEGER,
                prediction INTEGER,
                probability REAL
            )
        """)


def save_prediction(d: dict):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO predictions
            (timestamp, machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (d["timestamp"], d["machine_type"], d["air_temp"], d["process_temp"],
              d["rotational_speed"], d["torque"], d["tool_wear"], d["prediction"], d["probability"]))


def fetch_all():
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


app = FastAPI(title="Predictive Maintenance")
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")
templates = Jinja2Templates(directory=base_dir / "templates")

init_db()


def run_prediction(machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear):
    temp_diff = process_temp - air_temp
    encoded_type = type_map[machine_type]

    # prep data for model
    x = pd.DataFrame([{
        "Type": encoded_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature Difference": temp_diff,
    }])

    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    prob = round(float(model.predict_proba(x_scaled)[0][1]) * 100, 2)

    return pred, prob


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "type_options": list(type_map.keys()),
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    machine_type: str = Form(...),
    air_temp: float = Form(...),
    process_temp: float = Form(...),
    rotational_speed: int = Form(...),
    torque: float = Form(...),
    tool_wear: int = Form(...),
):
    errors = []
    if machine_type not in type_map:
        errors.append(
            f"Invalid machine type. Choose: {', '.join(type_map.keys())}")
    if not (290 <= air_temp <= 310):
        errors.append("Air temperature must be 290–310 K.")
    if not (295 <= process_temp <= 315):
        errors.append("Process temperature must be 295–315 K.")
    if not (1000 <= rotational_speed <= 3000):
        errors.append("Rotational speed must be 1000–3000 rpm.")
    if not (0 <= torque <= 100):
        errors.append("Torque must be 0–100 Nm.")
    if not (0 <= tool_wear <= 300):
        errors.append("Tool wear must be 0–300 min.")

    if not errors and process_temp <= air_temp:
        errors.append(
            "Process temperature must be greater than air temperature.")

    form = {
        "machine_type": machine_type, "air_temp": air_temp,
        "process_temp": process_temp, "rotational_speed": rotational_speed,
        "torque": torque, "tool_wear": tool_wear,
    }

    if errors:
        return templates.TemplateResponse(request, "index.html", {
            "type_options": list(type_map.keys()),
            "errors": errors,
            "form": form,
        })

    pred, prob = run_prediction(
        machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear
    )

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **form,
        "prediction": pred,
        "probability": prob,
    }
    save_prediction(record)

    return templates.TemplateResponse(request, "index.html", {
        "type_options": list(type_map.keys()),
        "result": record,
        "form": record,
    })


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    return templates.TemplateResponse(request, "history.html", {
        "rows": fetch_all(),
    })


@app.get("/export/csv")
async def export_csv():
    rows = fetch_all()
    if not rows:
        raise HTTPException(404, "No records found.")

    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    out.seek(0)

    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/export/pdf")
async def export_pdf():
    rows = fetch_all()
    if not rows:
        raise HTTPException(404, "No records found.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1*cm, rightMargin=1*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems = []

    # pdf title
    title_style = ParagraphStyle("t", parent=styles["Heading1"],
                                 fontSize=14, textColor=colors.HexColor("#1a365d"))
    elems.append(Paragraph("Predictive Maintenance — History", title_style))
    elems.append(Paragraph(
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Records: {len(rows)}",
        styles["Normal"]
    ))
    elems.append(Spacer(1, 0.4*cm))

    headers = ["#", "Timestamp", "Type", "Air T", "Proc T",
               "RPM", "Torque", "Wear", "Result", "Prob %"]

    data = [headers] + [[
        str(r["id"]), r["timestamp"], r["machine_type"],
        str(r["air_temp"]), str(r["process_temp"]), str(r["rotational_speed"]),
        str(r["torque"]), str(r["tool_wear"]),
        "FAILURE" if r["prediction"] == 1 else "OK",
        f"{r['probability']}%",
    ] for r in rows]

    col_w = [1*cm, 3.8*cm, 1.5*cm, 1.8*cm, 1.8 *
             cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.8*cm, 1.8*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)

    # format table rows
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a365d")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",        (0, 0), (-1, -1), 4),
    ]))

    for i, r in enumerate(rows, 1):
        if r["prediction"] == 1:
            t.setStyle(TableStyle([
                ("BACKGROUND", (8, i), (8, i), colors.HexColor("#fed7d7")),
                ("TEXTCOLOR",  (8, i), (8, i), colors.HexColor("#c53030")),
                ("FONTNAME",   (8, i), (8, i), "Helvetica-Bold"),
            ]))

    elems.append(t)
    doc.build(elems)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=predictions.pdf"},
    )

#!/usr/bin/env python3
"""Overlay tray-tilt CSVs produced by tray_tilt_eval_node.cpp.

Run tray_tilt_eval_node twice with identical parameters except for run_label /
hold_arm_fixed, then compare the two logs:

    python3 plot_tilt_comparison.py \
        tilt_eval_logs/tilt_eval_policy_<ts>.csv \
        tilt_eval_logs/tilt_eval_arm_fixed_<ts>.csv \
        -o tilt_comparison.png

Headless-friendly: uses the Agg backend and always saves a PNG instead of
calling plt.show().
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", help="One or more tilt_eval_*.csv files to overlay")
    parser.add_argument("-o", "--output", default="tilt_comparison.png", help="Output PNG path")
    args = parser.parse_args()

    runs = []
    for path in args.csv_files:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{path} has no rows -- was the run interrupted before any TF sample landed?")
        runs.append(df)

    # Crear dos subgráficas que comparten el eje X
    fig, (ax_tilt, ax_err) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 1. Graficar la inclinación de la bandeja (Tray Tilt)
    for df in runs:
        label = df["run_label"].iloc[0]
        ax_tilt.plot(df["t_sec"], df["tilt_deg"], label=label, linewidth=1.2)

    ax_tilt.set_ylabel("Tray tilt (deg)")
    ax_tilt.legend()
    ax_tilt.grid(True, alpha=0.3)

    # Obtener los límites de las fases de la primera ejecución para dibujar las líneas verticales
    ref = runs[0]
    boundaries = ref.loc[ref["phase_idx"].diff().fillna(0) != 0, "t_sec"]
    for t in boundaries:
        ax_tilt.axvline(t, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax_err.axvline(t, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    # 2. Graficar el error de posición del End-Effector en lugar de la velocidad de comandos
    for df in runs:
        label = df["run_label"].iloc[0]
        # Asegúrate de que tu nodo C++ haya guardado la columna como 'ef_pos_error'
        ax_err.plot(df["t_sec"], df["ef_pos_error"], label=f"Error {label}", linewidth=1.2)

    ax_err.set_xlabel("t (s)")
    ax_err.set_ylabel("Error de posición EF (m)")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    fig.suptitle("Desempeño de la Bandeja: Inclinación y Error de Posición")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

    # Resumen numérico rápido por ejecución y fase para ambas métricas
    for df in runs:
        label = df["run_label"].iloc[0]
        print(f"\n=== {label} ===")
        # Agrupamos también usando el error de posición para el resumen en consola
        summary = df.groupby("phase_name")[["tilt_deg", "ef_pos_error"]].agg(["mean", "std", "max"])
        print(summary.round(3))


if __name__ == "__main__":
    main()
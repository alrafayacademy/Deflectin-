import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# ---------- Beam deflection formulas for a simply-supported beam ----------
# Point load P at position a (distance from left support), span L
# For 0 <= x <= a:
#   y(x) = (P * b * x * (L**2 - b**2 - x**2)) / (6 * E * I * L)
# For a <= x <= L:
#   y(x) = (P * a * (L - x) * (L**2 - a**2 - (L - x)**2)) / (6 * E * I * L)
# where b = L - a
#
# UDL w (force per unit length) across full span:
#   y(x) = (w * x * (L**3 - 2*L*x**2 + x**3)) / (24 * E * I)
# max (center): delta_max = 5*w*L**4 / (384*E*I)

def deflection_point_load(x_array, L, P, a, E, I):
    """
    Computes deflection array for a point load P at position a on a simply-supported beam.
    x_array: numpy array of positions
    Returns y_array of deflections (same units as load*length^3/(E*I)).
    """
    b = L - a
    y = np.zeros_like(x_array, dtype=float)
    # Prevent division by zero for degenerate beam
    denom = 6 * E * I * L
    if denom == 0:
        return y * 0.0

    for i, x in enumerate(x_array):
        if x <= a:
            y[i] = (P * b * x * (L**2 - b**2 - x**2)) / denom
        else:
            # x >= a
            y[i] = (P * a * (L - x) * (L**2 - a**2 - (L - x)**2)) / denom
    return y

def deflection_udl(x_array, L, w, E, I):
    """
    Computes deflection array for a uniformly distributed load w (force/length)
    on a simply-supported beam.
    """
    denom = 24 * E * I
    if denom == 0:
        return np.zeros_like(x_array)
    x = x_array
    y = (w * x * (L**3 - 2 * L * x**2 + x**3)) / denom
    return y

# Helper to compute deflection and max values
def compute_deflection_and_plot(L, load_type, load_magnitude, load_pos, E, I, n_points=501):
    # Validation / safety
    if L <= 0:
        return None, "Error: Beam length L must be > 0."
    if E <= 0 or I <= 0:
        return None, "Error: E and I must be > 0."
    x = np.linspace(0, L, n_points)

    if load_type == "Point Load":
        P = load_magnitude
        a = float(np.clip(load_pos, 0.0, L))
        y = deflection_point_load(x, L, P, a, E, I)
        # For point load the standard max deflection formula at mid (if a == L/2) is P*L^3/(48EI).
        # But we will find numerical maximum from y array for general a.
    elif load_type == "UDL (full span)":
        # load_magnitude is w (force per unit length)
        w = load_magnitude
        y = deflection_udl(x, L, w, E, I)
    else:
        return None, "Error: Unsupported load type."

    # Deflection direction: formulas produce positive deflection (downwards). We'll keep sign but show absolute magnitude.
    max_deflection_idx = np.argmax(np.abs(y))
    max_deflection = y[max_deflection_idx]
    max_x = x[max_deflection_idx]

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.plot(x, y, linewidth=2)
    ax.axhline(0, linestyle='--', linewidth=0.8)
    ax.set_xlabel("Position along beam, x (same length units as L)")
    ax.set_ylabel("Deflection y (same length units as result)")
    ax.set_title(f"Beam deflection curve — {load_type}")
    # Mark the max deflection point
    ax.plot(max_x, max_deflection, marker='o')
    # Annotate numeric value
    ax.annotate(
        f"max deflection = {max_deflection:.6g} at x = {max_x:.6g}",
        xy=(max_x, max_deflection),
        xytext=(0.6 * L, 0.8 * np.min(y) if np.min(y) < 0 else 0.2 * np.max(y)),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.2)
    )
    ax.grid(True)

    # Flip y-axis sign if you'd rather show downward as negative (optional)
    # ax.invert_yaxis()

    # Return figure and result text
    result_text = (
        f"Computed on simply-supported beam.\n"
        f"Maximum deflection magnitude = {abs(max_deflection):.6g} (at x = {max_x:.6g}).\n"
        f"(Note: load magnitude interpreted as {'P (single force)' if load_type=='Point Load' else 'w (force per length)'}.)"
    )
    return fig, result_text

# ---------- Gradio UI ----------
with gr.Blocks(title="Beam Deflection Calculator") as demo:
    gr.Markdown("## Beam Deflection Calculator — Simply Supported Beam")
    gr.Markdown(
        "Enter beam and load parameters below. The app computes the deflection curve using standard Euler–Bernoulli formulas and shows the maximum deflection."
    )

    with gr.Row():
        with gr.Column(scale=1):
            L_input = gr.Slider(label="Beam length L", minimum=0.1, maximum=50.0, value=5.0, step=0.1)
            load_type = gr.Dropdown(choices=["Point Load", "UDL (full span)"], value="Point Load", label="Load type")
            load_magnitude = gr.Number(label="Load magnitude (P for point load, or w for UDL)", value=1000.0)
            load_pos = gr.Slider(label="Point load position a (distance from left)", minimum=0.0, maximum=5.0, value=2.5, step=0.01)
            E_input = gr.Number(label="Modulus of Elasticity E (e.g. 2.1e11 for steel, Pa)", value=2.1e11)
            I_input = gr.Number(label="Moment of Inertia I (e.g. m^4)", value=8.333e-6)
            compute_btn = gr.Button("Compute & Plot", variant="primary")

            gr.Examples(
                examples=[
                    # [L, load_type, load_magnitude, load_pos, E, I]
                    [5.0, "Point Load", 1000.0, 2.5, 2.1e11, 8.333e-6],
                    [4.0, "Point Load", 500.0, 2.0, 2.1e11, 1.0e-5],
                    [6.0, "UDL (full span)", 200.0, 3.0, 2.1e11, 1.2e-5],
                ],
                inputs=[L_input, load_type, load_magnitude, load_pos, E_input, I_input],
                label="Try example inputs"
            )

        with gr.Column(scale=1):
            plot_output = gr.Plot(label="Deflection curve (y vs x)")
            result_box = gr.Textbox(label="Result (max deflection)", interactive=False)

    # Update load_pos slider max whenever L changes
    def update_pos_slider(L):
        # returns properties: (min, max, value)
        return gr.update(max=L, value=round(L / 2, 4))

    L_input.change(fn=update_pos_slider, inputs=[L_input], outputs=[load_pos])

    # Main compute function wrapper
    def on_compute(L, load_type_sel, load_mag, load_position, E, I):
        # Prepare inputs
        try:
            L = float(L)
            E = float(E)
            I = float(I)
            load_mag = float(load_mag)
            load_position = float(load_position)
        except Exception as exc:
            return None, f"Error parsing inputs: {exc}"

        fig, result_text = compute_deflection_and_plot(L, load_type_sel, load_mag, load_position, E, I)
        if fig is None:
            return None, result_text
        return fig, result_text

    compute_btn.click(fn=on_compute,
                      inputs=[L_input, load_type, load_magnitude, load_pos, E_input, I_input],
                      outputs=[plot_output, result_box])

    gr.Markdown(
        "### Notes\n"
        "- This app models a simply-supported beam (two supports at ends).\n"
        "- For **Point Load**, the `Load magnitude` is the concentrated force P (N).\n"
        "- For **UDL**, the `Load magnitude` is the distributed load w (force per unit length, N/m).\n"
        "- Units must be consistent (e.g., meters for length, N for forces, Pa for E, m^4 for I).\n"
        "- The formulas used are standard Euler–Bernoulli beam formulas."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
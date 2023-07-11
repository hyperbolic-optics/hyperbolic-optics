from structure import Structure
import json
from payloads import updating_payload
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec
import tensorflow as tf
import numpy as np

tf.get_logger().setLevel("ERROR")


def mock_interface():
    def initial_data(axes):
        payload = json.loads(
            updating_payload("Incident", 5.5, 0.0, 0, 0, np.pi / 4.0, 475)
        )
        structure = Structure()
        structure.execute(payload)

        x_axis = np.round(np.degrees(structure.incident_angle), 1)
        frequency = structure.frequency.numpy().real

        reflectivities = [
            structure.r_pp,
            structure.r_ps,
            structure.r_sp,
            structure.r_ss,
        ]

        reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
        R_pp = reflectivities[0]
        R_ps = reflectivities[1]
        R_sp = reflectivities[2]
        R_ss = reflectivities[3]
        R_p_total = R_pp + R_ps
        R_s_total = R_ss + R_sp

        ax_to_plot = [
            (R_pp, "$|R_{pp}|^2$", axes[0]),
            (R_ps, "$|R_{ps}|^2$", axes[1]),
            (R_p_total, "$|R_{pp}|^2 + |R_{ps}|^2$", axes[2]),
            (R_sp, "$|R_{sp}|^2$", axes[3]),
            (R_ss, "$|R_{ss}|^2$", axes[4]),
            (R_s_total, "$|R_{ss}|^2 + |R_{sp}|^2$", axes[5]),
        ]

        return x_axis, frequency, ax_to_plot

    def scenario_handling(_):
        scenario_type = scenario_radio_buttons.value_selected

        if scenario_type == "Incident":
            x_label = "Incident Angle / $^\circ$"
            y_label = "$\omega/2\pi c (cm^{-1})$"

            # Show rotation Z slider and hide incident angle slider
            slider_incident_angle_ax.set_visible(False)
            slider_z_ax.set_visible(True)

        elif scenario_type == "Azimuthal":
            x_label = "Azimuthal Angle / $^\circ$"
            y_label = "$\omega/2\pi c (cm^{-1})$"

            # Show incident angle slider and hide rotation Z slider
            slider_incident_angle_ax.set_visible(True)
            slider_z_ax.set_visible(False)
            rotation_z_slider.set_val(0)

        else:
            x_label = "Azimuthal Angle / $^\circ$"
            y_label = "$\omega/2\pi c (cm^{-1})$"

        for axis in axes:
            axis.set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)

    def update(_):
        scenario_type = scenario_radio_buttons.value_selected
        eps_prism = eps_prism_slider.val
        air_gap_thickness = air_gap_thickness_slider.val
        rotation_y = rotation_y_slider.val
        rotation_z = rotation_z_slider.val
        incident_angle = incident_angle_slider.val

        payload = json.loads(
            updating_payload(
                scenario_type,
                eps_prism,
                air_gap_thickness,
                rotation_y,
                rotation_z,
                incident_angle,
                475,
            )
        )
        structure = Structure()
        structure.execute(payload)

        reflectivities = [
            structure.r_pp,
            structure.r_ps,
            structure.r_sp,
            structure.r_ss,
        ]

        reflectivities = np.round((reflectivities * np.conj(reflectivities)).real, 6)
        # reflectivities = np.round(np.asarray(reflectivities).imag, 6)
        R_pp = reflectivities[0]
        R_ps = reflectivities[1]
        R_sp = reflectivities[2]
        R_ss = reflectivities[3]
        R_p_total = R_pp + R_ps
        R_s_total = R_ss + R_sp

        ax_to_plot = [
            (R_pp, "$|R_{pp}|^2$", axes[0]),
            (R_ps, "$|R_{ps}|^2$", axes[1]),
            (R_p_total, "$|R_{pp}|^2 + |R_{ps}|^2$", axes[2]),
            (R_sp, "$|R_{sp}|^2$", axes[3]),
            (R_ss, "$|R_{ss}|^2$", axes[4]),
            (R_s_total, "$|R_{ss}|^2 + |R_{sp}|^2$", axes[5]),
        ]

        for i, (reflectivity, title, axis) in enumerate(ax_to_plot):
            im = axis.collections[0]
            im.set_array(reflectivity.ravel())
            im.set_clim(vmin=0, vmax=reflectivity.max())
            colorbar_list[i].mappable.set_clim(
                0,
            )  # Update the colorbar limits correctly
            colorbar_list[i].draw_all()

        plt.draw()

    width, height = figaspect(6.0 / 4.0)

    fig = plt.figure(figsize=(width, height))
    grid = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(grid[i, j]))

    x_axis, frequency, ax_to_plot = initial_data(axes)

    colorbar_list = []
    for data, title, axis in ax_to_plot:
        im = axis.pcolormesh(x_axis, frequency, data, cmap="magma")
        cbar = plt.colorbar(im, ax=axis)
        colorbar_list.append(cbar)
        cbar.mappable.set_clim(
            0,
        )
        cbar.set_label(title)
        axis.set_title(title)
        axis.set_xticks(np.linspace(x_axis.min(), x_axis.max(), 5))
        axis.set_xlabel("Incident Angle / $^\circ$")
        axis.set_ylabel("$\omega/2\pi c (cm^{-1})$")

    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.25, top=0.95, hspace=0.5, wspace=0.4
    )

    ### Parameter Components

    ## Scenario Radio Buttons
    scenario_radio_ax = plt.axes([0.01, 0.01, 0.1, 0.15])
    scenario_radio_buttons = RadioButtons(
        scenario_radio_ax, ("Incident", "Azimuthal", "Dispersion"), active=0
    )

    ## Plot-Style Radio Buttons
    plot_style_radio_ax = plt.axes([0.12, 0.01, 0.1, 0.15])
    plot_style_radio_buttons = RadioButtons(
        plot_style_radio_ax, ("Real", "Imaginary", "Absolute"), active=0
    )

    ## Slider Bars

    slider_thickness_ax = plt.axes([0.28, 0.13, 0.5, 0.025])
    slider_eps_prism_ax = plt.axes([0.28, 0.09, 0.5, 0.025])
    slider_y_ax = plt.axes([0.28, 0.05, 0.5, 0.025])

    air_gap_thickness_slider = Slider(slider_thickness_ax, "Air Gap", 0, 1.5, valinit=0)
    eps_prism_slider = Slider(
        slider_eps_prism_ax, f"$\epsilon_p$", 4.5, 6.5, valinit=5.5
    )
    rotation_y_slider = Slider(slider_y_ax, "Rotation Y", 0, 90, valinit=0)

    slider_incident_angle_ax = plt.axes([0.28, 0.01, 0.5, 0.025], visible=False)
    incident_angle_slider = Slider(
        slider_incident_angle_ax, "Incident Angle", 0, 90, valinit=0
    )

    slider_z_ax = plt.axes([0.28, 0.01, 0.5, 0.025], visible=True)
    rotation_z_slider = Slider(slider_z_ax, "Rotation Z", 0, 90, valinit=0)

    ## Checkboxes and Save Button
    subplot_labels = [
        "$|r_{pp}|^2$",
        "$|r_{ps}|^2$",
        "$|r_{pp}|^2 + |r_{ps}|^2$",
        "$|r_{sp}|^2$",
        "$|r_{ss}|^2$",
        "$|r_{ss}|^2 + |r_{sp}|^2$",
    ]

    checkboxes_ax = []
    subplot_checkboxes = []
    for i in range(2):
        for j in range(3):
            checkboxes_ax.append(
                fig.add_axes([0.82 + i * 0.07, 0.06 + j * 0.03, 0.07, 0.03])
            )  # Adjust these numbers to properly place checkboxes
            subplot_checkboxes.append(
                CheckButtons(checkboxes_ax[-1], [subplot_labels[i * 3 + j]], [False])
            )

    save_button_ax = plt.axes([0.82, 0.02, 0.14, 0.04])
    save_button = Button(save_button_ax, "SAVE")

    air_gap_thickness_slider.on_changed(update)
    eps_prism_slider.on_changed(update)
    rotation_y_slider.on_changed(update)
    rotation_z_slider.on_changed(update)
    incident_angle_slider.on_changed(update)

    scenario_radio_buttons.on_clicked(scenario_handling)

    scenario_radio_buttons.on_clicked(update)
    plot_style_radio_buttons.on_clicked(update)

    update(None)

    plt.show()
    plt.close()


mock_interface()

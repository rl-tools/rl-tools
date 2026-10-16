CONTROLS = ("motor", "ctbr", "position")
ACTUATIONS = ("rotors", "ideal")


def validate_control(control, actuation):
    if control not in CONTROLS:
        raise ValueError("hyperdrone: control must be 'motor', 'ctbr' or 'position'")
    if actuation not in ACTUATIONS:
        raise ValueError("hyperdrone: actuation must be 'rotors' or 'ideal'")
    if control == "ctbr" and actuation == "rotors":
        raise ValueError("hyperdrone: control='ctbr' on actuation='rotors' needs the PD rate controller's gains, which the presets do not carry; use actuation='ideal'")
    if control != "ctbr" and actuation == "ideal":
        raise ValueError(f"hyperdrone: actuation='ideal' takes thrust and body-rate commands; control={control!r} emits motor commands, use control='ctbr'")

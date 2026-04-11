"""
Runner for all Petrophysics December 2023 (Vol 64, No 6) article modules.

References (SPWLA Petrophysics Vol 64, No 6, December 2023):
  - Bennis et al.,    pp. 931-953  (radial Sw inversion)
  - Bradley et al.,   pp. 823-836  (universal wellbore data format)
  - Cely et al.,      pp. 919-930  (mud-gas viscosity)
  - Garcia et al.,    pp. 879-889  (2D NMR Gaussian decomposition)
  - Khan et al.,      pp. 954-969  (salt-cavern creep damage)
  - McGlynn et al.,   pp. 900-918  (pulsed-neutron C/O)
  - Trevizan & Menezes de Jesus, pp. 890-899 (GAN image-log SR)
  - Wang & Ehlig-Economides,    pp. 970-977 (CO2 solubility)
"""
import importlib

MODULES = [
    "bennis_invasion_sw",
    "bradley_wellbore_format",
    "cely_mudgas_viscosity",
    "garcia_nmr_gaussian",
    "khan_salt_creep",
    "mcglynn_pulsed_neutron",
    "trevizan_gan_image_log",
    "wang_co2_solubility",
]


def test_all():
    failed = []
    for name in MODULES:
        print(f"\n=== {name} ===")
        try:
            m = importlib.import_module(name)
            m.test_all()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(name)
    print("\n" + "=" * 40)
    print(f"{len(MODULES) - len(failed)}/{len(MODULES)} modules passed")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    test_all()

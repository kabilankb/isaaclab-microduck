#!/usr/bin/env python
"""Convert the Microduck MJCF models to USD for Isaac Lab / Newton.

Newton builds its physics model from USD (`ModelBuilder.add_usd`), so USD is the
only asset path into Isaac Lab — the MJCFs cannot be loaded directly. This drives
Isaac Sim's MJCF importer (`isaacsim.asset.importer.mjcf`, the `mujoco-usd-converter`
backend from Isaac Sim 5.0 on) over every model in `isaaclab_microduck.assets`.

    python scripts/convert_assets.py                    # all models, lazily
    python scripts/convert_assets.py --model walk --force

Output: `isaaclab_microduck/assets/usd/<stem>/<stem>.usda` plus a `payloads/`
directory (gitignored build artifacts — regenerate, never hand-edit).

Conversion is lazy: an existing USD is reused unless `--force` is given. Note the
importer does NOT notice edited mesh files, only an edited MJCF, so pass `--force`
after touching anything under `assets/`.

Verify the result with `check_asset_parity.py` — conversion silently dropping or
reinterpreting a property is the main risk this whole phase exists to catch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_microduck.assets import ROBOT_MJCF, USD_DIR, mjcf_path, usd_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=sorted(ROBOT_MJCF), help="Convert one model (default: all).")
parser.add_argument("--force", action="store_true", help="Re-convert even if the USD already exists.")
parser.add_argument(
    "--no-self-collision",
    action="store_true",
    help="Disable self-collision between links (default: enabled).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402


def convert(model_name: str) -> str:
    """Convert one model and return the path of the generated USD interface file.

    Raises if the importer wrote somewhere other than the expected location. It
    does that more readily than you would think: when the output directory already
    holds an asset it cannot reuse, it writes a SIBLING ``<stem>_1/`` and reports
    success, leaving the stale original in place for everything else to load. A
    silently stale robot is the worst possible failure mode here, so treat a
    surprise path as an error rather than trusting the return value.
    """
    source = mjcf_path(model_name)
    expected = usd_path(model_name)
    if args_cli.force and expected.parent.is_dir():
        shutil.rmtree(expected.parent)
    cfg = MjcfConverterCfg(
        asset_path=str(source),
        usd_dir=str(USD_DIR),
        force_usd_conversion=args_cli.force,
        # Self-collision matters here: the `self_collisions` penalty is a real term
        # in the Microduck reward stack, and the full-collision models exist
        # precisely so limbs can hit the trunk.
        self_collision=not args_cli.no_self_collision,
        # The MJCFs carry explicit `class="collision"` geometry; do not synthesise
        # collision from the visual meshes.
        collision_from_visuals=False,
        # MuJoCo collides mesh geoms as convex hulls, so this matches the source.
        collision_type="Convex Hull",
        # Keep meshes separate — merged meshes lose the per-geom identity that the
        # contact sensors and the `.*_collision` selectors rely on.
        merge_mesh=False,
        # Timestep/gravity come from our own SimulationCfg, not the MJCF.
        import_physics_scene=False,
        # Floating base: the trunk carries a freejoint.
        fix_base=False,
    )
    converter = MjcfConverter(cfg)
    produced = Path(converter.usd_path).resolve()
    if produced != expected.resolve():
        raise RuntimeError(
            f"Importer wrote '{produced}' but '{expected}' was expected. A stale asset in "
            f"the output directory usually causes this — re-run with --force."
        )
    return str(produced)


def flatten_rigid_body_hierarchy(usd_file: str) -> int:
    """Make nested rigid bodies ignore their ancestors' transforms.

    USD physics writes a WORLD-space transform to every rigid body prim. The MuJoCo
    USD Converter preserves MuJoCo's nested ``<body>`` tree (14 of our 15 bodies sit
    inside another body), where transforms are RELATIVE -- so in USD each nested
    body gets its parent's world transform composed on top of its own and the robot
    renders EXPLODED, displacement growing with depth. NVIDIA's own UR10e ships 9
    rigid bodies with 0 nested; siblings are the convention.

    Physics is unaffected, which is why this survived so long unnoticed: Newton
    builds its model from the joint graph, the Newton viewer draws from Newton's own
    shape transforms, and `check_asset_parity.py` compares physics. Only a
    USD-native renderer (Kit/RTX) shows the damage.

    Rather than reparent -- which would mean consistent namespace surgery across the
    four layers these prims compose from (physx/physics/base/robot), or flattening
    away the instancing that keeps 4096 envs affordable -- each nested body gets
    ``!resetXformStack!``, the USD op that drops ancestor transforms. Its local
    transform is re-authored to its former WORLD transform so the rest pose is
    unchanged, and everything is written as OVERRIDES in the root layer, leaving the
    converter's payloads untouched.

    Returns the number of bodies fixed.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_file, Usd.Stage.LoadAll)
    bodies = [pr for pr in stage.Traverse() if pr.HasAPI(UsdPhysics.RigidBodyAPI)]

    def is_nested(prim):
        par = prim.GetParent()
        while par and par.IsValid() and not par.IsPseudoRoot():
            if par.HasAPI(UsdPhysics.RigidBodyAPI):
                return True
            par = par.GetParent()
        return False

    nested = [b for b in bodies if is_nested(b)]
    if not nested:
        return 0

    # World transforms captured BEFORE any edit; these are what must survive.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world = {b.GetPath(): cache.GetLocalToWorldTransform(b) for b in nested}

    root = stage.GetRootLayer()
    with Usd.EditContext(stage, root):
        for body in nested:
            xf = UsdGeom.Xformable(stage.OverridePrim(body.GetPath()))
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
            op.Set(world[body.GetPath()])
            # "!resetXformStack!" first: ignore every ancestor transform, then apply
            # this body's own world matrix.
            xf.SetXformOpOrder([op], resetXformStack=True)

    root.Save()
    return len(nested)


def main() -> None:
    models = [args_cli.model] if args_cli.model else sorted(ROBOT_MJCF)
    print(f"[convert_assets] MJCF source: {mjcf_path(models[0]).parent}")
    print(f"[convert_assets] USD output : {USD_DIR}\n")

    failures: list[tuple[str, Exception]] = []
    for model_name in models:
        try:
            usd = convert(model_name)
            moved = flatten_rigid_body_hierarchy(usd)
            if moved:
                print(f"[convert_assets]   flattened {moved} nested rigid bod(y/ies)")
            print(f"  OK   {model_name:24s} -> {usd}")
        except Exception as exc:  # noqa: BLE001 - report every model, fail at the end
            failures.append((model_name, exc))
            print(f"  FAIL {model_name:24s} -> {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)}/{len(models)} model(s) failed to convert.")
        sys.exit(1)
    print(f"\n{len(models)} model(s) converted.")


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Isaac Sim extension: Microduck policy playback, registered in the Examples Browser.

Follows Isaac Sim 6.0.1's own example pattern (`isaacsim.examples.interactive.hello_world`):
a `BaseSampleUITemplate` handing Load / Reset / Clear, registered through
`isaacsim.examples.browser`. Registering with the browser is also what makes the
extension discoverable without fighting `--ext-folder`.

NOTE ON VERSIONS. Isaac Sim 4.5 exposed these as
`isaacsim.examples.interactive.base_sample`; 6.0.1 moved them to
`isaacsim.examples.base.base_sample_experimental`. This targets 6.0.1 -- the imports
are guarded so an unsupported version says so instead of failing obscurely at startup.
"""

from __future__ import annotations

import os

import omni.ext
import omni.ui as ui

from .sample import MicroduckPolicySample

_EXAMPLE_NAME = "Microduck Policy Player"
_CATEGORY = "Policy"


class MicroduckPolicyPlayerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self.example_name = _EXAMPLE_NAME
        self.category = _CATEGORY
        try:
            from isaacsim.examples.base.base_sample_extension_experimental import (
                BaseSampleUITemplate,
            )
            from isaacsim.examples.browser import get_instance as get_browser_instance
        except ImportError as exc:  # pragma: no cover - version guard
            import carb

            carb.log_error(
                f"[microduck.policy.player] Isaac Sim example API not found ({exc}). "
                "This extension targets Isaac Sim 6.0.1; on 4.5 the module is "
                "isaacsim.examples.interactive.base_sample."
            )
            return

        self._sample = MicroduckPolicySample()
        ui_handle = _MicroduckUI(
            ext_id=ext_id,
            file_path=os.path.abspath(__file__),
            title="Microduck Policy Player",
            doc_link="https://github.com/kabilankb/isaaclab-microduck",
            overview=(
                "Load a trained Microduck policy (exported/policy.pt) and run it on a "
                "Microduck in this stage. The export has the observation normalizer "
                "baked in; a raw model_*.pt checkpoint does not and will misbehave."
            ),
            sample=self._sample,
            template_cls=BaseSampleUITemplate,
        )
        self._ui_handle = ui_handle
        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self) -> None:
        try:
            from isaacsim.examples.browser import get_instance as get_browser_instance

            get_browser_instance().deregister_example(
                name=self.example_name, category=self.category
            )
        except Exception:  # pragma: no cover - teardown is best-effort
            pass


class _MicroduckUI:
    """Wraps BaseSampleUITemplate and appends the policy controls.

    Composition rather than subclassing, so the template class can be passed in and the
    4.5 / 6.0.1 import difference stays confined to the extension's startup.
    """

    def __init__(self, template_cls, sample, **kwargs) -> None:
        self._sample = sample
        self._template = template_cls(sample=sample, **kwargs)
        self._policy_field = None
        self._speed_field = None
        self._status = None

    def build_ui(self):
        self._template.build_ui()
        self._build_policy_frame()

    def _build_policy_frame(self) -> None:
        frame = ui.CollapsableFrame(title="Policy", height=0)
        with frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Exported policy (run dir or exported/policy.pt)", height=18)
                with ui.HStack(spacing=4, height=24):
                    self._policy_field = ui.StringField()
                    ui.Button("Load", width=70, clicked_fn=self._on_load)
                with ui.HStack(spacing=4, height=24):
                    ui.Label("Forward command (m/s)", width=160)
                    self._speed_field = ui.FloatField()
                    self._speed_field.model.set_value(0.0)
                    ui.Button("Set", width=60, clicked_fn=self._on_set_speed)
                self._status = ui.Label(self._sample.status, word_wrap=True, height=40)

    def _on_load(self) -> None:
        path = self._policy_field.model.get_value_as_string().strip()
        if not path:
            self._set("Enter a path to an exported policy.")
            return
        try:
            self._sample.load_policy(path)
        except Exception as exc:
            self._set(f"ERROR: {exc}")
            return
        self._set(self._sample.status)

    def _on_set_speed(self) -> None:
        vx = self._speed_field.model.get_value_as_float()
        self._sample.set_forward_command(vx)
        self._set(f"Forward command set to {vx:.2f} m/s.")

    def _set(self, text: str) -> None:
        if self._status is not None:
            self._status.text = text

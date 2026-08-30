"""Kit UI: attach to a running Microduck env and hot-swap the policy driving it.

This panel does NOT create the environment -- see `runner.py` for why that is
impossible from inside a Kit extension (SimulationContext is a singleton, so a
newly-made env silently discards the task's Newton config). Launch the session with
`scripts/play.py --visualizer kit`, then attach from here.

Thin layer: everything touching the env or the policy lives in `runner.py`, which
imports nothing from `omni.*` and is therefore testable without Kit.
"""

from __future__ import annotations

import omni.ext
import omni.ui as ui

from .runner import PolicyDriver, PolicyRunnerError, list_play_tasks

_HELP = (
    "Pick a task and press Create to build it here.\n"
    "If a play.py session is already running in this process, press Attach instead --"
    " a second env would inherit its simulation context and ignore the task's Newton"
    " config."
)

#: Env counts offered in the UI. Small by design: this is for WATCHING. The same cfg
#: that converged at 4096 envs failed completely at 32, so playback counts must never
#: be read as evidence about a policy.
ENV_COUNTS = [1, 2, 4, 9]


class MicroduckPolicyPlayerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._driver = PolicyDriver()
        self._sub = None
        self._playing = False
        self._build_window()

    def on_shutdown(self) -> None:
        self._stop_stepping()
        self._driver.detach()
        if getattr(self, "_window", None) is not None:
            self._window.destroy()
            self._window = None

    # -- UI ---------------------------------------------------------------

    def _build_window(self) -> None:
        self._window = ui.Window("Microduck Policy Player", width=560, height=400)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Task", height=18)
                try:
                    self._tasks = list_play_tasks()
                except Exception as exc:
                    self._tasks = []
                    self._set_status(f"Could not list tasks: {exc}", error=True)
                self._task_combo = ui.ComboBox(0, *self._tasks)

                with ui.HStack(spacing=4, height=24):
                    ui.Label("Environments", width=90)
                    self._envs_combo = ui.ComboBox(2, *[str(n) for n in ENV_COUNTS])

                with ui.HStack(spacing=4, height=28):
                    ui.Button("Create env", clicked_fn=self._on_create)
                    ui.Button("Attach to running", clicked_fn=self._on_attach)
                    ui.Button("Release", width=80, clicked_fn=self._on_detach)
                self._env_label = ui.Label("No environment.", word_wrap=True, height=32)

                ui.Label("Exported policy (run directory or exported/policy.pt)", height=18)
                with ui.HStack(spacing=4, height=24):
                    self._path_field = ui.StringField()
                    ui.Button("Load", width=70, clicked_fn=self._on_load_policy)

                with ui.HStack(spacing=4, height=28):
                    self._play_button = ui.Button("Play", clicked_fn=self._on_toggle_play)
                    ui.Button("Reset", clicked_fn=self._on_reset)

                self._status = ui.Label(_HELP, word_wrap=True, height=72)
                self._steps = ui.Label("steps: 0", height=18)

    def _set_status(self, text: str, error: bool = False) -> None:
        if getattr(self, "_status", None) is not None:
            self._status.text = ("ERROR: " + text) if error else text

    # -- actions ----------------------------------------------------------

    def _selected(self, combo, values):
        idx = combo.model.get_item_value_model().as_int
        return values[idx] if 0 <= idx < len(values) else None

    def _on_create(self) -> None:
        task = self._selected(self._task_combo, self._tasks)
        if not task:
            self._set_status("No task selected. Is isaaclab_microduck importable here?", error=True)
            return
        n = int(self._selected(self._envs_combo, [str(x) for x in ENV_COUNTS]) or 4)
        self._stop_stepping()
        self._set_status(f"Creating {task} with {n} envs. This can take a minute...")
        try:
            desc = self._driver.create(task, num_envs=n)
        except PolicyRunnerError as exc:
            self._set_status(str(exc), error=True)
            return
        except Exception as exc:
            self._set_status(f"Unexpected failure: {exc}", error=True)
            return
        self._env_label.text = f"Created: {desc}"
        self._set_status("Environment created. Load an exported policy.")

    def _on_attach(self) -> None:
        try:
            desc = self._driver.attach()
        except PolicyRunnerError as exc:
            self._set_status(str(exc), error=True)
            return
        self._env_label.text = f"Attached: {desc}"
        self._set_status("Attached. Load an exported policy.")

    def _on_detach(self) -> None:
        self._stop_stepping()
        self._driver.detach()
        self._env_label.text = "No environment."
        self._steps.text = "steps: 0"
        self._set_status(_HELP)

    def _on_load_policy(self) -> None:
        path = self._path_field.model.get_value_as_string().strip()
        if not path:
            self._set_status("Enter a path to an exported policy.", error=True)
            return
        try:
            self._driver.set_policy(path)
        except PolicyRunnerError as exc:
            self._set_status(str(exc), error=True)
            return
        self._set_status("Policy loaded. Press Play.")

    def _on_toggle_play(self) -> None:
        if not self._driver.is_ready:
            self._set_status("Attach to an env and load a policy first.", error=True)
            return
        if self._playing:
            self._stop_stepping()
            self._set_status("Paused.")
        else:
            self._start_stepping()
            self._set_status("Playing.")

    def _on_reset(self) -> None:
        try:
            self._driver.reset()
        except Exception as exc:
            self._set_status(f"Reset failed: {exc}", error=True)
            return
        self._steps.text = "steps: 0"
        self._set_status("Reset.")

    # -- stepping ---------------------------------------------------------

    def _start_stepping(self) -> None:
        import omni.kit.app

        if self._sub is not None:
            return
        self._sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="microduck_policy_player")
        )
        self._playing = True
        self._play_button.text = "Pause"

    def _stop_stepping(self) -> None:
        self._sub = None  # dropping the subscription unsubscribes
        self._playing = False
        if getattr(self, "_play_button", None) is not None:
            self._play_button.text = "Play"

    def _on_update(self, _event) -> None:
        """One control step per Kit frame.

        Stepping from the update stream rather than a loop: a `while` loop inside an
        extension freezes the viewport it is meant to render. An exception here would
        fire every frame, so a failure stops playback instead of flooding the log.
        """
        try:
            self._driver.step()
        except Exception as exc:
            self._stop_stepping()
            self._set_status(f"Stopped after an error: {exc}", error=True)
            return
        if self._driver.steps % 10 == 0:
            self._steps.text = f"steps: {self._driver.steps}"

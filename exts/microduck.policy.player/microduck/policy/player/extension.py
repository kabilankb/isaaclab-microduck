"""Kit UI for running a trained Microduck policy in the Isaac Sim viewport.

Thin layer: everything that touches the env or the policy lives in `runner.py`, which
imports nothing from `omni.*` and is therefore testable without Kit.

Stepping is driven from the Kit update loop rather than a `while` loop, because a busy
loop inside an extension freezes the UI and the viewport along with it.
"""

from __future__ import annotations

import omni.ext
import omni.ui as ui

from .runner import PolicyRunner, PolicyRunnerError, list_play_tasks

#: Env counts offered in the UI. Small by design -- this is for WATCHING. The same cfg
#: that converged at 4096 envs failed completely at 32, so playback counts must never
#: be read as training results.
ENV_COUNTS = [1, 4, 9, 16]


class MicroduckPolicyPlayerExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._runner = PolicyRunner()
        self._playing = False
        self._sub = None
        self._tasks: list[str] = []
        self._build_window()

    def on_shutdown(self) -> None:
        self._stop_stepping()
        self._runner.unload()
        if getattr(self, "_window", None) is not None:
            self._window.destroy()
            self._window = None

    # -- UI ---------------------------------------------------------------

    def _build_window(self) -> None:
        self._window = ui.Window("Microduck Policy Player", width=460, height=340)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Task", height=18)
                try:
                    self._tasks = list_play_tasks()
                except Exception as exc:  # registration can fail if the pkg is absent
                    self._tasks = []
                    self._set_status(f"Could not list tasks: {exc}", error=True)
                self._task_combo = ui.ComboBox(0, *self._tasks)

                ui.Label("Exported policy (run directory or exported/policy.pt)", height=18)
                with ui.HStack(spacing=4, height=24):
                    self._path_field = ui.StringField()
                    ui.Button("Browse", width=70, clicked_fn=self._on_browse)

                with ui.HStack(spacing=4, height=24):
                    ui.Label("Environments", width=90)
                    self._envs_combo = ui.ComboBox(1, *[str(n) for n in ENV_COUNTS])

                with ui.HStack(spacing=4, height=28):
                    ui.Button("Load", clicked_fn=self._on_load)
                    self._play_button = ui.Button("Play", clicked_fn=self._on_toggle_play)
                    ui.Button("Reset", clicked_fn=self._on_reset)
                    ui.Button("Unload", clicked_fn=self._on_unload)

                self._status = ui.Label("Idle.", word_wrap=True, height=40)
                self._steps = ui.Label("steps: 0", height=18)

    def _set_status(self, text: str, error: bool = False) -> None:
        if getattr(self, "_status", None) is None:
            return
        self._status.text = ("ERROR: " + text) if error else text

    def _selected(self, combo, values):
        idx = combo.model.get_item_value_model().as_int
        return values[idx] if 0 <= idx < len(values) else None

    # -- actions ----------------------------------------------------------

    def _on_browse(self) -> None:
        try:
            from omni.kit.window.filepicker import FilePickerDialog
        except ImportError:
            self._set_status("File picker unavailable; type the path instead.", error=True)
            return

        def _picked(filename: str, dirname: str) -> None:
            self._path_field.model.set_value(f"{dirname}/{filename}" if filename else dirname)
            dialog.hide()

        dialog = FilePickerDialog(
            "Select exported policy",
            apply_button_label="Select",
            click_apply_handler=_picked,
        )

    def _on_load(self) -> None:
        task = self._selected(self._task_combo, self._tasks)
        if not task:
            self._set_status("No task selected. Is isaaclab_microduck importable?", error=True)
            return
        path = self._path_field.model.get_value_as_string().strip()
        if not path:
            self._set_status("Choose an exported policy first.", error=True)
            return
        n = int(self._selected(self._envs_combo, [str(x) for x in ENV_COUNTS]) or 4)

        self._stop_stepping()
        self._set_status(f"Loading {task} ({n} envs)...")
        try:
            self._runner.load(task, path, num_envs=n)
        except PolicyRunnerError as exc:
            self._set_status(str(exc), error=True)
            return
        except Exception as exc:  # pragma: no cover - surfaced to the user
            self._set_status(f"Unexpected failure: {exc}", error=True)
            return
        self._set_status(f"Loaded {task}. Press Play.")

    def _on_toggle_play(self) -> None:
        if not self._runner.is_loaded:
            self._set_status("Load a policy first.", error=True)
            return
        if self._playing:
            self._stop_stepping()
            self._set_status("Paused.")
        else:
            self._start_stepping()
            self._set_status("Playing.")

    def _on_reset(self) -> None:
        self._runner.reset()
        self._steps.text = "steps: 0"
        self._set_status("Reset.")

    def _on_unload(self) -> None:
        self._stop_stepping()
        self._runner.unload()
        self._steps.text = "steps: 0"
        self._set_status("Unloaded.")

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

        A raised exception here would fire every frame, so failures stop playback
        instead of flooding the log.
        """
        try:
            self._runner.step()
        except Exception as exc:
            self._stop_stepping()
            self._set_status(f"Stopped after an error: {exc}", error=True)
            return
        if self._runner.steps % 10 == 0:
            self._steps.text = f"steps: {self._runner.steps}"

"""Custom environment that visualizes ground contact by changing body colors to red."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class ContactVisRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with visual contact feedback on robot bodies.

    When a monitored body contacts the ground (force > threshold), its visual mesh
    turns red. When contact ends or the episode resets, the original color is restored
    by removing the override material binding.
    """

    CONTACT_THRESHOLD = 1.0  # Newtons
    # Only visualize unexpected contacts (exclude feet which always touch ground)
    VIS_BODY_NAMES = {"base_link", "LL_hip_yaw", "LL_hip_roll", "LL_hip_pitch", "LL_knee",
                      "RL_hip_yaw", "RL_hip_roll", "RL_hip_pitch", "RL_knee"}

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._contact_vis_initialized = False
        self._body_names: list[str] | None = None

    def _init_contact_vis(self):
        """Lazy initialization of contact visualization."""
        if self._contact_vis_initialized:
            return

        try:
            from pxr import Gf, Sdf, UsdShade

            import omni.usd

            stage = omni.usd.get_context().get_stage()
            self._stage = stage

            # Create a single red OmniPBR material
            red_mat_path = "/World/Looks/ContactRedMaterial"
            if not stage.GetPrimAtPath(red_mat_path).IsValid():
                UsdShade.Material.Define(stage, red_mat_path)
                shader = UsdShade.Shader.Define(stage, red_mat_path + "/Shader")
                shader.CreateIdAttr("OmniPBR")
                shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(1.0, 0.0, 0.0)
                )
                shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.5)
                mat = UsdShade.Material.Get(stage, red_mat_path)
                mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

            self._red_mat_path = red_mat_path

            # Get body names from contact sensor
            contact_sensor = self.scene.sensors["contact_forces"]
            self._body_names = list(contact_sensor.body_names)
            self._num_bodies = len(self._body_names)

            # Filter to only monitored bodies
            self._vis_body_indices = [i for i, name in enumerate(self._body_names)
                                      if name in self.VIS_BODY_NAMES]

            # Build visual prim path templates (from env_0)
            self._body_visual_paths: dict[int, str] = {}  # body_idx -> visuals prim path
            for body_idx in self._vis_body_indices:
                body_name = self._body_names[body_idx]
                vis_path = f"/World/envs/env_0/Robot/{body_name}/visuals"
                if stage.GetPrimAtPath(vis_path).IsValid():
                    self._body_visual_paths[body_idx] = vis_path

            # Track contact state
            self._prev_contact_state = torch.zeros(
                self.num_envs, self._num_bodies, dtype=torch.bool, device=self.device
            )

            self._contact_vis_initialized = True
            vis_names = [self._body_names[i] for i in self._body_visual_paths]
            print(f"[ContactVisRLEnv] Initialized: monitoring {vis_names}")

        except Exception as e:
            import traceback
            print(f"[ContactVisRLEnv] Failed to initialize: {e}")
            traceback.print_exc()
            self._contact_vis_initialized = True
            self._body_names = None

    def step(self, action: torch.Tensor):
        """Override step to add contact color visualization."""
        result = super().step(action)

        if self.sim.has_gui():
            self._init_contact_vis()

            # Restore colors for environments that were just reset
            reset_env_ids = (self.episode_length_buf == 1).nonzero(as_tuple=False).squeeze(-1).tolist()
            if reset_env_ids:
                self._reset_contact_colors(reset_env_ids)

            self._update_contact_colors()

        return result

    def _bind_red(self, env_id: int, body_idx: int):
        """Bind red material to a body's visuals."""
        from pxr import UsdShade

        tmpl_path = self._body_visual_paths.get(body_idx)
        if not tmpl_path:
            return
        prim_path = tmpl_path.replace("/env_0/", f"/env_{env_id}/")
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            red_mat = UsdShade.Material.Get(self._stage, self._red_mat_path)
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(red_mat, UsdShade.Tokens.strongerThanDescendants)

    def _unbind(self, env_id: int, body_idx: int):
        """Remove material override to restore original appearance."""
        from pxr import UsdShade

        tmpl_path = self._body_visual_paths.get(body_idx)
        if not tmpl_path:
            return
        prim_path = tmpl_path.replace("/env_0/", f"/env_{env_id}/")
        prim = self._stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.UnbindDirectBinding()

    def _reset_contact_colors(self, env_ids: list[int]):
        """Restore original colors for reset environments."""
        if not self._body_names:
            return
        for env_id in env_ids:
            for body_idx in self._vis_body_indices:
                if self._prev_contact_state[env_id, body_idx]:
                    self._unbind(env_id, body_idx)
                    self._prev_contact_state[env_id, body_idx] = False

    def _update_contact_colors(self):
        """Check body contacts and update colors."""
        if not self._body_names:
            return

        try:
            contact_sensor = self.scene.sensors["contact_forces"]
            net_forces = contact_sensor.data.net_forces_w
            force_magnitudes = torch.norm(net_forces, dim=-1)
            contact_mask = force_magnitudes > self.CONTACT_THRESHOLD

            changed = contact_mask != self._prev_contact_state
            if not changed.any():
                return

            env_ids, body_ids = changed.nonzero(as_tuple=True)

            for env_id, body_id in zip(env_ids.tolist(), body_ids.tolist()):
                if body_id not in self._body_visual_paths:
                    continue

                if contact_mask[env_id, body_id]:
                    self._bind_red(env_id, body_id)
                else:
                    self._unbind(env_id, body_id)

            self._prev_contact_state = contact_mask.clone()

        except Exception as e:
            import traceback
            print(f"[ContactVisRLEnv] Error: {e}")
            traceback.print_exc()
            self._body_names = None
